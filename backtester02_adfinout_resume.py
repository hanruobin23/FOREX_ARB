from cmath import polar
from re import X
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as smlm
import statsmodels.regression.rolling as smr
import statsmodels.tools.tools as smt
import scipy.optimize as so
import scipy.stats as ss
import pytz
import datetime as dt
import numpy_ext
from numba import njit
import itertools
from joblib import Parallel, delayed
import functools
plt.rcParams.update({'font.size': 14,'lines.markersize':2,'figure.constrained_layout.use':True,'savefig.dpi':300})


#All data before 2019 is sparse: for accurate testing, begin on start of 2019 = timestamp 1546293360 for both 1 min and 5 min bars
#IDEAS: determine thresholds for trade by identifying clusters of turning points
#scp /Users/Ruobin/Desktop/QUANT_TRADING/FOREX_pairs/backtester02_adfinout.py grace:/home/rh848/palmer_scratch/python/FOREX_pairs
#============================CODE BEGINS=====================================
# ticker: labels for what I'm trading. Could be a real symbol, could be a 'mock' ticker
# bar: either 'm1' (1 minute bars) or 'm5' (5 minute bars)
# window: number of bars to consider in calculating adfuller p-values and z scores (2880: 10 trading days)
def gen_adfuller_zscore_all(bar,window):
    # Runs gen_adfuller_zscore() for all tickers
    # An example of the format of data is found in the directory .\data
    tickers=[filename.split('_')[0] for filename in os.listdir(os.path.join(os.getcwd(),'data')) if f'{bar}' in filename]
    meta_dir=os.path.join(os.getcwd(),f'{bar}_meta_data')
    if not os.path.isdir(meta_dir):
        os.makedirs(meta_dir)
    else:
        done_tickers=np.unique([filename.split('_')[0] for filename in os.listdir(os.path.join(os.getcwd(),f'{bar}_meta_data'))])
        for ticker in done_tickers:
            if ticker in tickers:
                tickers.remove(ticker)  
    for ticker in tickers:
        print(ticker)
        gen_adfuller_zscore(ticker,bar,window)                                                  

def gen_adfuller_zscore(ticker,bar,window):
    # Calculates the adfuller p-value and z_score for one ticker at each timestamp and saves the calculation results. These values will be used repeatedly during the optimization process.
    output_dir=os.path.join(os.getcwd(),f'{bar}_meta_data')
    def rolling_adfuller_zscore(ticker_arr):
        return [adfuller(ticker_arr)[1],ss.zscore(ticker_arr)[-1]]
    #assume fixed number of bars per window (e.g. a week is extended to 8 days in a window if one of the days is a public hol)
    start_time=1546293360
    data_dir=os.path.join(os.getcwd(),'data')
    #maybe should not .dropna() since window size will be slightly affected 
    df=pd.read_csv(os.path.join(data_dir,f'{ticker}_{bar}.csv'),index_col=None,header=0).rename({'close':ticker},axis=1).set_index('time')[[ticker]]
    df=df.loc[df.index.values>=start_time].dropna()
    df[[f'adfuller_pval_{window}',f'zscore_{window}']]=numpy_ext.rolling_apply(rolling_adfuller_zscore,window,df[ticker].to_numpy(),n_jobs=-1)
    df[[f'adfuller_pval_{window}',f'zscore_{window}']].to_csv(os.path.join(output_dir,f'{ticker}_{window}.csv'))
    return

@njit(parallel=False)
def trade_signals(exr_arr,adf_arr,z_arr,p_in,p_out,long_in,long_out,short_in,short_out):
    # INPUTS:
    # exr_arr: 1D array of exchange rates (prices)
    # adf_arr: 1D array of adfuller p-values
    # z_arr: 1D array of z_scores
    # p_in,p_out,long_in,long_out,short_in,short_out: 6 parameters controlling trade
    # returns: calculated as a fraction of margin used for this position
    # TRADE LOGIC:
    # if in a long position: 
    #   if pval>p_out:
    #       close long position
    #   elif zscore>long_out:
    #       close long position
    #       if zscore>short_in:
    #           open short position
    # elif in a short position: 
    #   if pval>p_out:
    #       close short position
    #   elif zscore<short_out:
    #       close short position
    #       if zscore<long_in:
    #           open long position
    # elif not in a position and pval<p_in:
    #   if zscore<long_in:
    #       open long position
    #   if zscore>short_in:
    #       open short position
    # STOP LOSS IMPLEMENTATION:
    # I allow my trades to lose as much as 100% before closing. The allowed drift in exr is thus 1/leverage*entry exr.
    # NOTE:Ideally I would be using tick data / data at higher frequencies to model realistic stop loss implementation. However, this would significantly slow down the entire backtest process, so an approximation using the same 5 minute bars is used.
    # After long/short stop loss is triggered, I pause future opening of long/short positions until one of the following conditions are met:
    # - enable opening of long and short: series becomes non-stationary (exit from current cycle)
    # - enable opening of long: zscore>long_out (exit from current trough to mean movement)
    # - enable opening of short: zscore<short_out (exit from current peak to mean movement)
    leverage=500
    # long_signal: saves the points in time when either an in or out signal is generated for a long position, along with the corresponding returns on each trade
    long_signal=np.empty(len(adf_arr))
    long_signal[:]=np.nan
    # longs: saves the points in time when a long position is held, along with the corresponding returns on each trade
    longs=np.empty(adf_arr.shape)
    longs[:]=np.nan
    short_signal=np.empty(adf_arr.shape)
    short_signal[:]=np.nan
    shorts=np.empty(adf_arr.shape)
    shorts[:]=np.nan
    # adf_out: saves the points in time when a position is closed due to non-stationarity of price series
    adf_out=np.empty(adf_arr.shape)
    adf_out[:]=np.nan
    # long_stop: saves the points in time when a stop loss is called for a long position
    long_stop=np.empty(adf_arr.shape)
    long_stop[:]=np.nan
    short_stop=np.empty(adf_arr.shape)
    short_stop[:]=np.nan
    # curr_status: saves the current position of the ticker. 0: no position, 1: long, -1: short.
    curr_status=0
    long_returns=[]
    short_returns=[]
    # entrance: saves the exchange rate at which the position is opened. Initial value is unimportant.
    entrance=1
    # lin: saves the index along the input arrays at which a long position is opened
    lin=0
    sin=0
    # lstop: saves the exchange rate below which a stop loss is called for the current open position.
    lstop=0
    sstop=0
    # lpause: indicates whether to pause opening a long position due to a recent closure from a stop loss.
    lpause=False
    spause=False
    for i in range(len(adf_arr)):
        if not np.isnan(adf_arr[i]):
            if adf_arr[i]>p_out:
                lpause=False
                spause=False
            elif z_arr[i]>=long_out:
                lpause=False
            elif z_arr[i]<=short_out:
                spause=False
            if curr_status==1:
                if adf_arr[i]>p_out:
                    returns=(exr_arr[i]-entrance)*leverage/entrance
                    long_signal[lin]=returns
                    long_signal[i]=returns
                    longs[lin:i+1]=returns
                    long_returns.append(returns)
                    curr_status=0
                    adf_out[i]=1
                else:
                    if z_arr[i]>=long_out or exr_arr[i]<=lstop:
                        returns=(exr_arr[i]-entrance)*leverage/entrance
                        long_signal[lin]=returns
                        long_signal[i]=returns
                        longs[lin:i+1]=returns
                        long_returns.append(returns)
                        if exr_arr[i]<lstop:
                            long_stop[i]=1
                            lpause=True
                        if z_arr[i]>=short_in and spause==False:
                            sin=i
                            entrance=exr_arr[i]
                            curr_status=-1
                            sstop=(1+1/leverage)*entrance
                        else:
                            curr_status=0
            elif curr_status==-1:
                if adf_arr[i]>p_out:
                    returns=(entrance-exr_arr[i])*leverage/entrance
                    short_signal[sin]=returns
                    short_signal[i]=returns
                    shorts[sin:i+1]=returns
                    short_returns.append(returns)
                    curr_status=0
                    adf_out[i]=1
                else:
                    if z_arr[i]<=short_out or exr_arr[i]>=sstop:
                        returns=(entrance-exr_arr[i])*leverage/entrance
                        short_signal[sin]=returns
                        short_signal[i]=returns
                        shorts[sin:i+1]=returns
                        short_returns.append(returns)
                        if exr_arr[i]>sstop:
                            short_stop[i]=1
                            spause=True
                        if z_arr[i]<=long_in and lpause==False:
                            lin=i
                            entrance=exr_arr[i]
                            curr_status=1
                            lstop=(1-1/leverage)*entrance
                        else:
                            curr_status=0
            elif curr_status==0:
                if adf_arr[i]<=p_in:
                    if z_arr[i]<=long_in and lpause==False:
                        lin=i
                        entrance=exr_arr[i]
                        curr_status=1
                    elif z_arr[i]>=short_in and spause==False:
                        sin=i
                        entrance=exr_arr[i]
                        curr_status=-1
    # Closes position on the last datapoint
    if curr_status==1:
        returns=(exr_arr[-1]-entrance)*leverage/entrance
        long_signal[lin]=returns
        long_signal[-1]=returns
        longs[lin:]=returns
        long_returns.append(returns)
    elif curr_status==-1:
        returns=(entrance-exr_arr[-1])*leverage/entrance
        short_signal[sin]=returns
        short_signal[-1]=returns
        shorts[sin:]=returns
        short_returns.append(returns)
    return (long_signal,longs,short_signal,shorts,adf_out,long_stop,short_stop,np.array(long_returns),np.array(short_returns))                    

def run_trade(ticker,bar,window,thresholds):
    # Running this with a given set of parameters will output a graph plotting each trade returns against time. Green will represent a long position, while red represents a short position.
    data_dir=os.path.join(os.getcwd(),'data')
    meta_dir=os.path.join(os.getcwd(),f'{bar}_meta_data')
    fig_dir=os.path.join(os.getcwd(),'figures')
    p_in,p_out,long_in,long_out,short_in,short_out=thresholds
    adf_z_df=pd.read_csv(os.path.join(meta_dir,f'{ticker}_{window}.csv'),index_col=None,header=0).set_index('time')
    tickers_df=pd.read_csv(os.path.join(data_dir,f'{ticker}_{bar}.csv')).set_index('time')[['close']].rename({'close':ticker},axis=1)
    df=pd.concat([tickers_df,adf_z_df],axis=1)
    adf_arr=df[f'adfuller_pval_{str(window)}'].values
    z_arr=df[f'zscore_{str(window)}'].values   
    exr_arr=df[ticker].values
    long_signal,longs,short_signal,shorts,adf_out,long_stop,short_stop,long_returns,short_returns=trade_signals(exr_arr,adf_arr,z_arr,p_in,p_out,long_in,long_out,short_in,short_out)
    df[['long_signal','longs','short_signal','shorts','adf_out','long_stop','short_stop']]=np.vstack((long_signal,longs,short_signal,shorts,adf_out,long_stop,short_stop)).T
    df[['long_signal','longs','short_signal','shorts','adf_out','long_stop','short_stop']].to_csv(os.path.join(meta_dir,f'{ticker}_{window}_trades.csv'))
    overall_returns=np.hstack((long_returns,short_returns))
    adf_out_num=np.count_nonzero(~np.isnan(adf_out))
    long_stop_num=np.count_nonzero(~np.isnan(long_stop))
    short_stop_num=np.count_nonzero(~np.isnan(short_stop))
    print('LONG===========')
    print('MEAN:',np.mean(long_returns))
    print('MEDIAN:',np.median(long_returns))
    print('STANDARD DEVIATION:',np.std(long_returns))
    print('LONG TRADE#:',len(long_returns))
    print('LONG WIN RATE:',np.sum(long_returns>0)/np.sum(long_returns<0))
    print('LONG STOP OUT#:',long_stop_num)
    print('SHORT===========')
    print('MEAN:',np.mean(short_returns))
    print('MEDIAN:',np.median(short_returns))
    print('STANDARD DEVIATION:',np.std(short_returns))
    print('SHORT TRADE#:',len(short_returns))
    print('SHORT WIN RATE:',np.sum(short_returns>0)/np.sum(short_returns<0))
    print('SHORT STOP OUT#:',short_stop_num)
    print('OVERALL===========')
    print('MEAN:',np.mean(overall_returns))
    print('MEDIAN:',np.median(overall_returns))
    print('STANDARD DEVIATION:',np.std(overall_returns))
    print('TOTAL TRADE#:',len(overall_returns))
    print('OVERALL WIN RATE:',np.sum(overall_returns>0)/np.sum(overall_returns<0))
    print('TOTAL STOP#:',long_stop_num+short_stop_num)
    print('TOTAL ADF OUT#:',adf_out_num)
    plt.figure()
    plt.plot(df.index.values,df['long_signal'],'og')
    plt.plot(df.index.values,df['short_signal'],'or')
    plt.savefig(os.path.join(fig_dir,f'{ticker}_{bar}.png'))
    plt.close()
    return

def get_returns_std_num_total(input,exr_arr,adf_arr,z_arr):
    p_in=input[0]
    p_out=input[1]
    long_in=input[2]
    long_out=input[3]
    short_in=input[4]
    short_out=input[5]
    long_signal,longs,short_signal,shorts,adf_out,long_stop,short_stop,long_returns,short_returns=trade_signals(exr_arr,adf_arr,z_arr,p_in,p_out,long_in,long_out,short_in,short_out)
    overall_returns=np.hstack((long_returns,short_returns))
    if overall_returns.size==0:
        return(0,0,0,0)
    return (np.mean(overall_returns),np.std(overall_returns),len(overall_returns),np.sum(overall_returns))

def gen_grid_optim(ticker,bar,window):
    # writes a csv of total returns, std of trade returns, mean returns, number of trades for each combination of the 6 parameters
    data_dir=os.path.join(os.getcwd(),'data')
    meta_dir=os.path.join(os.getcwd(),f'{bar}_meta_data')
    grid_eval_dir=os.path.join(os.getcwd(),f'grid_eval_adfinout_{bar}_{window}')
    adf_z_df=pd.read_csv(os.path.join(meta_dir,f'{ticker}_{window}.csv'),index_col=None,header=0).set_index('time')
    tickers_df=pd.read_csv(os.path.join(data_dir,f'{ticker}_{bar}.csv')).set_index('time')
    if 'close' in tickers_df.columns:
        tickers_df=tickers_df[['close']].rename({'close':ticker},axis=1)
    df=pd.concat([tickers_df,adf_z_df],axis=1)
    adf_arr=df[f'adfuller_pval_{str(window)}'].values
    z_arr=df[f'zscore_{str(window)}'].values   
    exr_arr=df[ticker].values
    # Creates a list of all possible combinations of 6 parameters given the step size
    thresholds=list(itertools.product([0.005+i*0.005 for i in range(10)],[0.05+i*0.005 for i in range(10)],[-2.0+i*0.1 for i in range(10)],[-0.5+i*0.05 for i in range(10)],[1.0+i*0.1 for i in range(10)],[0.05+i*0.05 for i in range(10)]))
    # Runs the grid evaluation using multiprocessing
    grid_eval=Parallel(n_jobs=-1)(delayed(functools.partial(get_returns_std_num_total,exr_arr=exr_arr,adf_arr=adf_arr,z_arr=z_arr))(threshold) for threshold in thresholds)
    threshold_df=pd.DataFrame(thresholds,columns=['p_in','p_out','long_in','long_out','short_in','short_out'])
    grid_eval_df=pd.DataFrame(grid_eval,columns=['overall_mean','overall_std','overall_trade#','overall_returns'])
    pd.concat([threshold_df,grid_eval_df],axis=1).to_csv(os.path.join(grid_eval_dir,f'{ticker}_{bar}.csv'))
    return

def gen_grid_optim_all(bar,window):
    # Generates the grid csv file for each ticker
    meta_dir=os.path.join(os.getcwd(),f'{bar}_meta_data')
    grid_eval_dir=os.path.join(os.getcwd(),f'grid_eval_adfinout_{bar}_{window}')
    if not os.path.isdir(grid_eval_dir):
        os.makedirs(grid_eval_dir)
    tickers=[filename.split('_')[0] for filename in os.listdir(meta_dir) if '.DS' not in filename and '-' not in filename and 'trades' not in filename]
    done_tickers=[filename.split('_')[0] for filename in os.listdir(grid_eval_dir)]
    for ticker in done_tickers:
        if ticker in tickers:
            tickers.remove(ticker)
    for ticker in tickers:
        print(ticker)
        gen_grid_optim(ticker,bar,window)
    return       

def get_optim_params_from_grid(bar,window):
    # Extracts the optimal combination of parameters from each ticker and outputs into a file. Also obtains the statistics (e.g. mean and std) for the span of the parameter space for each ticker.
    # NOTE: Here I assumed that slippage due to spread could be about 10%, so I included this in my calculation for the total returns. This is purely to penalise parameters which yielded high total returns with numerous small trades.
    grid_eval_dir=os.path.join(os.getcwd(),f'grid_eval_adfinout_{bar}_{window}')
    optim_df_list=[]
    for f in os.listdir(grid_eval_dir):
        if '.DS_Store' not in f:
            df=pd.read_csv(os.path.join(grid_eval_dir,f),header=0,index_col=0)
            #assume slippage results in 10% loss
            df['overall_returns']=(df['overall_mean']-0.1)*df['overall_trade#']
            optim_row=df.loc[df[['overall_returns']].idxmax()].copy()
            worst_row=df.loc[df[['overall_returns']].idxmin()].copy()
            optim_row=optim_row.rename(index={optim_row.index.values[0]:f.split('_')[0]}).rename(columns={'overall_returns':'optim_returns','overall_mean':'optim_mean','overall_std':'optim_std','overall_trade#':'optim_trade#'})
            optim_row[['pessim_mean','pessim_std','pessim_trade#','pessim_returns']]=worst_row[['overall_mean','overall_std','overall_trade#','overall_returns']].values[0]
            optim_row['trades_returns_mean']=np.average(df['overall_mean'].values,weights=df['overall_trade#'].values)
            optim_row['param_returns_mean']=np.mean(df['overall_returns'].values)
            optim_row['param_returns_std']=np.std(df['overall_returns'].values)
            optim_row['param_mean_mean']=np.mean(df['overall_mean'].values)
            optim_row['param_mean_std']=np.std(df['overall_mean'].values)
            optim_row['param_std_mean']=np.mean(df['overall_std'].values)
            optim_row['param_std_std']=np.std(df['overall_std'].values)
            optim_df_list.append(optim_row)
    pd.concat(optim_df_list).to_csv(f'optim_params_adfinout_{bar}_{window}.csv')
    return

# Running the following will yield my results for all tradeable tickers.
# I have not yet included the function for generating mock data; once you have the mock data, put them in a separate directory from your real data, and run everything with different directories (meta,grid_eval).
# It is important to separate real data from mock data because the actual implementation of trading mock tickers is a little more tedious than trading real tickers.
if __name__ == '__main__':
    # gen_adfuller_zscore_all('m5',200)
    gen_grid_optim_all('m5',200)
    get_optim_params_from_grid('m5',200)