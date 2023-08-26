# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:15:27 2022

@author: hanruobin

"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
from statsmodels.tsa.stattools import adfuller
import scipy.stats as ss
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.combining import OrTrigger
from apscheduler.triggers.cron import CronTrigger
import datetime as dt
pd.set_option('display.max_columns', None)
# TERMINOLOGY=============
# SYMBOL: tradeable symbols for which a position can be opened/closed e.g. USDRUB
# TICKER: self-defined 'symbols' used to refer to either a pair or single trades. e.g. RUBXAG is not a tradeable symbol on my broker, but I refer to my trade pair of USDRUB and XAGUSD as RUBXAG. I might also be trading XAGUSD by itself, in which case XAGUSD is also a ticker.
# ========================
# FUTURE DEVELOPMENTS: Definitely should migrate to MQL5 framework to minimize time taken to poll bars
# FUTURE DEVELOPMENTS: save current bars and mock ticker prices, and create a rolling window that just needs to be updated with the latest bar.

login='<>'
pwd='<>'
svr='live.mt5tickmill.com'

forex_dir=os.getcwd()
# RISK MANAGEMENT: Up to 3% equity allcoated per ticker, up to 99% equity on all positions (margin requirement 0.2%)
max_size_proportion=0.03
run_code=True
# Initialization and login to connect to the MT5 terminal.
# Visit https://www.mql5.com/en/docs/integration/python_metatrader5/mt5login_py for more information
if not mt5.initialize('C:/Program Files/MetaTrader 5/terminal64.exe',
                      login=login,
                      password=pwd,
                      server=svr):
    print("initialize() failed")
    print(mt5.last_error())
    mt5.shutdown()
else:
    print('initialize() succeeded')

if not mt5.login(login=login,
                      password=pwd,
                      server=svr):
    print("login() failed for account #{}, error code:{}".format(login, mt5.last_error()))
    print(mt5.last_error())
    mt5.shutdown()
else:
    print('login() succeeded')
    acct_info=mt5.account_info()
    if acct_info is None:
        print(f'error obtaining account info. Error code: {mt5.last_error()}')
        mt5.shutdown()
    else:
        # Leverage required for a variety of calculations, including lot size & stop loss. Leverage=500 on my trading account
        leverage=acct_info.leverage
        print('Leverage:',leverage)

# I pick only the tickers that have a positive mean return-per-trade averaged across the 6D parameter space.
# From testing on the demo account, it turns out that one trading cycle takes about 20 seconds to complete, a significant lag relative to the size of my tick bars.
# Since I have some tickers that do better on average than others, I want the effect of lag to be minimized for the better performing ones, thus the df.sort_values()

optim_params_df=pd.read_csv(os.path.join(forex_dir,'optim_params_adfinout_m5_2880.csv'),header=0,index_col=0)
optim_params_mock_df=pd.read_csv(os.path.join(forex_dir,'optim_params_adfinout_mock_m5_2880.csv'),header=0,index_col=0)
optim_params_all_df=pd.concat([optim_params_mock_df,optim_params_df]).sort_values(by=['param_mean_mean'])
trade_tickers_df=optim_params_all_df.loc[optim_params_all_df['param_mean_mean']>0]
trade_tickers=list(trade_tickers_df.index.values)

# I'm listing all the symbols with 'USD' since I'll be using only them for trading mock tickers and calculate lot sizes 
all_avail_USD_symbols=[symbol for symbol in optim_params_df.index.values if 'USD' in symbol]

# Here I use a dictionary to record the 6 parameters (p_in, p_out, long_in, long_out, short_in, short_out) that will control the opening/closing of positions for each ticker
# I also want to know the 'type' of each ticker. If each ticker is 'real' (i.e. listed directly by my broker --> tradeable symbol) OR 'mock' (e.g. RUBXAG is traded with USDRUB and XAGUSD). The mechanical procedures for sending orders for these two types are a little different

# if type=='real' --> pairs==np.nan and pair_types==np.nan
# otherwise if type=='mock' (e.g. RUBXAG) --> pairs== 'USDRUB,XAGUSD' and pair_types=='s,s'.
# pair_types always match '(s|l),(s|l)'. 's' --> short, 'l' --> long, with respect to longing the ticker. (e.g. to long RUBXAG, I short both USDRUB and XAGUSD). This also tells me whether to invert the exchange rate when calculating the lot_sizes for each symbol for a 'mock' ticker.
trade_tickers_dict={}
for ticker in trade_tickers:
    ticker_dict={}
    for col in ['p_in','p_out','long_in','long_out','short_in','short_out','param_mean_mean','type','pairs','pair_types']:
        ticker_dict[col]=trade_tickers_df.loc[ticker][col]
    if ticker_dict['type']=='real':
        if ticker[:3]+'USD' in all_avail_USD_symbols:
            ticker_dict['symbol_lot']=ticker[:3]+'USD'
            ticker_dict['invert']=False
            # lot size = lot_value (in USD) / exchange rate (exr). Suppose we are looking at EURCHF, then the relevant exr is EURUSD ~ 1.1.
            # With 100 USD (already scaled by leverage), I can buy a lot size of 100/1.1/100000. Here, I divide by the 'uninverted' exr.
            # If the relevant exr is USDEUR instead (~0.9), the lot size will be 100/(1/0.9)/100000. Here, I divide by the 'inverted' exr.
        else:
            ticker_dict['symbol_lot']='USD'+ticker[:3]
            ticker_dict['invert']=True
    trade_tickers_dict[ticker]=ticker_dict

# 'lpause','spause' --> My strategy enforces additional conditions on entering a position if a stop loss was recently hit, so I want to record these occurrences. lpause==True --> stop loss on a long position occurred previously, spause==True --> '' short ''

# curr_pos is the current position of the ticker. I follow my broker's/mt5's convention: 0->long     1->short    2(self-defined)->none

# I keep track of which positions correspond to which ticker via 'positions'. Each position is a named_tuple returned by mt5. They hold information about the current position type (0 or 1), volume and ticket number. These are all required for closing my position.
# This is because I am trading symbols with 'USD' for my mock tickers, so there is a chance I will open more than one position for the same symbol.
# E.g. a long position on RUBXAG and TRYXAG both involve shorting XAGUSD. I want to keep track of which positions correspond to which ticker so they can be closed correctly. 

# ticket_nums is more of a temporary variable. To poll for the positions, I need to know the ticket number, which is first generated when placing an order. During order execution, I want to minimize time wasted on polling for the position since I may have more orders to execute.
# Furthermore, it might be some time before the order is actually executed, so there is a chance the poll returns an error. The solution is to simply keep track of the ticket number and poll the position at a later time.
# FUTURE IDEAS: volume and position types are all known before an order is placed, so they could probably be saved directly to the dictionary instead of named_tuples.
curr_status_filedir=os.path.join(os.getcwd(),'curr_status_m5.csv')
if not os.path.isfile(curr_status_filedir):
    status_dict={}
    for ticker in trade_tickers:
        status_dict[ticker]={'lpause':False,'spause':False,'curr_pos':2,'ticket_nums':[],'positions':[]}
else:
    curr_status_df=pd.read_csv(curr_status_filedir,index_col=0,header=0,dtype={'ticket_nums':str,'lpause':np.bool8,'spause':np.bool8,'curr_pos':np.int8}).fillna('')
    status_dict=curr_status_df.to_dict('index')
    for ticker in status_dict.keys():
        status_dict[ticker]['ticket_nums']=[] if status_dict[ticker]['ticket_nums']=='' else [int(ticket_num) for ticket_num in status_dict[ticker]['ticket_nums'].split(',')]
        status_dict[ticker]['positions']=[]
        for ticket_num in status_dict[ticker]['ticket_nums']:
            mt5_position=mt5.positions_get(ticket=ticket_num)
            if mt5_position is None:
                print(f'{ticker} failed to obtain position(s). Error code: {mt5.last_error()}')
                break
            position=mt5_position[0]
            status_dict[ticker]['positions'].append(position)
# This variable is here to ensure that check_sl() does not collide with trade(), since both modify status_dict
in_trade=False

def close_position(ticker,close_reason):
    global status_dict
    write_lines=[]
    for tradeposition in status_dict[ticker]['positions']:
        if tradeposition.type==1:
            order_type=mt5.ORDER_TYPE_BUY
        else:
            order_type=mt5.ORDER_TYPE_SELL
        close_ticket=tradeposition.ticket
        close_symbol=tradeposition.symbol
        request={
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": close_symbol,
            "volume": tradeposition.volume,
            "type": order_type,
            "position": close_ticket,
            "comment": f"{close_reason} {close_ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            }
        order=mt5.order_send(request)
        if order.retcode!=10009 and order.retcode!=10008:
            write_lines.append(f'{close_symbol} failed to close due to error: {order.retcode} {order.comment}')
            continue
        else:
            write_lines.append(f'{close_symbol} closed successfully. Ticket: {close_ticket}')
            status_dict[ticker]['curr_pos']=2
            status_dict[ticker]['ticket_nums']=[]
            status_dict[ticker]['positions']=[]
    return write_lines

def open_real(order_type,ticker,symbol_lot,ticker_invert,expected_trade_returns):
    global status_dict
    write_lines=[]
    acct_info=mt5.account_info()
    if acct_info is None:
        write_lines.append(f'{ticker} error retrieving account info for trade. Error code: {mt5.last_error()}')
        return write_lines
    # Calculates the maximum lot_val that satisfies both the 3% equity limit and 1% equity for margin requirements
    lot_val=leverage*min([acct_info.equity*max_size_proportion,acct_info.margin_free-acct_info.equity*0.01])
    tick=mt5.symbol_info_tick(symbol_lot)
    if tick is None:
        write_lines.append(f'{ticker} error retrieving lot size conversion info for {symbol_lot}. Error code: {mt5.last_error()}')
        return write_lines
    # divisor is the exr used to calculate lot size.
    # e.g. to short EURCHF, even though the symbol is directly tradeable, my funds are in USD. Thus, it is equivalent to shorting EURUSD and longing USDCHF.
    # Shorting EURUSD means taking the exr at the bid price (the highest price offered by buyers), while longing it means taking the exr at ask price.
    if order_type==mt5.ORDER_TYPE_SELL:
        if ticker_invert==False:
            divisor=tick.bid
        else:
            divisor=1/tick.ask
    else:
        if ticker_invert==False:
            divisor=tick.ask
        else:
            divisor=1/tick.bid
    #Price of XAUUSD or XAGUSD is given in USD/ounce of gold or silver, and 1 lot of XAUUSD is 100 ounces, 1 lot of XAGUSD is 5000 ounces
    if ticker=='XAUUSD':
        divisor=divisor/1000
    elif ticker=='XAGUSD':
        divisor=divisor/20
    
    #The ideal procedure would be to round off lot_val/divisor/100000, but since round() only rounds off to integers and rounds down for decimals, the factor of 100 is applied after the rounding
    lot_size=round(lot_val/divisor/1000)/100
    if lot_size==0:
        write_lines.append(f'{ticker} failed to open due to allocated lot size = 0')
        return write_lines
    tick=mt5.symbol_info_tick(ticker)
    if tick is None:
        write_lines.append(f'{ticker} error retrieving tick for spread calculation. Error code: {mt5.last_error()}')
        return write_lines
    else:
        total_spread_pct_lev=2*leverage*((tick.ask-tick.bid)/tick.bid+2/100000)
        if total_spread_pct_lev>expected_trade_returns:
            write_lines.append(f'{ticker} open (sell) failed because spread is too high. Estimated buy-sell spread_pct_lev: {total_spread_pct_lev}; Expected trade returns: {expected_trade_returns}')
            return write_lines
        else:
            if order_type==mt5.ORDER_TYPE_SELL:
                order_comment='os'
            else:
                order_comment='ob'
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": ticker,
                "volume":lot_size,
                "type": order_type,
                "comment": f"{order_comment} {ticker}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                }
            order=mt5.order_send(request)
            if order.retcode!=10009 and order.retcode!=10008:
                write_lines.append(f'{ticker} failed to open due to error: {order.retcode} {order.comment}')
                return write_lines
            else:
                order_num=order.order
                if order_type==mt5.ORDER_TYPE_SELL:
                    write_lines.append(f'{ticker} opened (sell) successfully. Ticket: {order_num}')
                    status_dict[ticker]['curr_pos']=1
                    status_dict[ticker]['ticket_nums'].append(order_num)
                else:
                    write_lines.append(f'{ticker} opened (buy) successfully. Ticket: {order_num}')
                    status_dict[ticker]['curr_pos']=0
                    status_dict[ticker]['ticket_nums'].append(order_num)
    return write_lines

def open_mock(order_type,ticker,symbol_pairs,symbol_pair_types,expected_trade_returns):
    # Since I am trading two symbols instead of one, I half the allocation to each symbol so that my exposure to this ticker is kept within 3%.
    global status_dict
    write_lines=[]
    acct_info=mt5.account_info()
    if acct_info is None:
        write_lines.append(f'{ticker} error retrieving account info for trade. Error code: {mt5.last_error()}')
        return write_lines
    lot_val=leverage*min([acct_info.equity*max_size_proportion,acct_info.margin_free-acct_info.equity*0.01])/2
    order_types=[]
    lot_sizes=[]
    total_spreads_pct_lev=[]
    divisors=[]
    # iterating through the pair of symbols to be traded
    for i in range(2):
        symbol_lot=symbol_pairs[i]
        tick=mt5.symbol_info_tick(symbol_lot)
        if tick is None:
            write_lines.append(f'Error retrieving lot size conversion info for {symbol_lot}. Error code: {mt5.last_error()}')
            break
        if order_type==mt5.ORDER_TYPE_SELL:
            if i==0:
                if symbol_pair_types[i]=='l':
                    divisor=tick.bid
                    order_types.append(mt5.ORDER_TYPE_SELL)
                else:
                    divisor=1
                    order_types.append(mt5.ORDER_TYPE_BUY)
            else:
                if symbol_pair_types[i]=='l':
                    divisor=1
                    order_types.append(mt5.ORDER_TYPE_SELL)
                else:
                    divisor=tick.ask
                    order_types.append(mt5.ORDER_TYPE_BUY)
        else:
            if i==0:
                if symbol_pair_types[i]=='l':
                    divisor=tick.ask
                    order_types.append(mt5.ORDER_TYPE_BUY)
                else:
                    divisor=1
                    order_types.append(mt5.ORDER_TYPE_SELL)
            else:
                if symbol_pair_types[i]=='l':
                    divisor=1
                    order_types.append(mt5.ORDER_TYPE_BUY)
                else:
                    divisor=tick.bid
                    order_types.append(mt5.ORDER_TYPE_SELL)
        if symbol_lot=='XAUUSD':
            divisor=divisor/1000
        elif symbol_lot=='XAGUSD':
            divisor=divisor/20
        lot_size=round(lot_val/divisor/1000)/100
        if lot_size==0:
            write_lines.append(f'{symbol_lot} failed to open for {ticker} due to allocated lot size = 0.')
            return write_lines
        divisors.append(divisor)
        lot_sizes.append(lot_size)
        total_spreads_pct_lev.append(2*leverage*((tick.ask-tick.bid)/tick.bid+2/100000))
    if len(lot_sizes)<2:
        return write_lines
    # I assume that the allocation to each symbol is the same (not yet verified), so the estimated spread of trading this mock ticker is the average of the two symbols'
    total_spread_pct_lev=np.mean(total_spreads_pct_lev)
    if total_spread_pct_lev>expected_trade_returns:
        write_lines.append(f'{ticker} open (sell) failed because spread is too high. Estimated buy-sell spread_pct_lev: {total_spread_pct_lev}; Expected trade returns: {expected_trade_returns}')
        return write_lines
    else:
        if divisors!=[1,1]:
            if divisors[0]>divisors[1]:
                scale_fac=divisors[0]/divisors[1]
                lot_sizes[1]=round(lot_sizes[0]*100*scale_fac)/100
            else:
                scale_fac=divisors[1]/divisors[0]
                lot_sizes[0]=round(lot_sizes[1]*100*scale_fac)/100
        if order_type==mt5.ORDER_TYPE_SELL:
            order_comment='os'
        else:
            order_comment='ob'
        for i in range(2):
            symbol=symbol_pairs[i]
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_sizes[i],
                "type": order_types[i],
                "comment": f"{order_comment} {ticker}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                }
            order=mt5.order_send(request)
            if order.retcode!=10009 and order.retcode!=10008:
                write_lines.append(f'{symbol} failed to open for {ticker} (sell) due to error: MT5-->{mt5.last_error()}, ORDER-->{order.retcode} {order.comment}')
                if order.retcode==10019:
                    write_lines.append(f'{symbol} no money')
                    check_order=mt5.order_check(request)
                    if check_order is None:
                        write_lines.append(f'{symbol} failed to check order due to error: {mt5.last_error()}')
                    else:
                        check_order_dict=check_order._asdict()
                        for key in check_order_dict:
                            write_lines.append(f'{key}: {check_order_dict[key]}')
                continue
            else:
                order_num=order.order
                if order_type==mt5.ORDER_TYPE_SELL:
                    write_lines.append(f'{symbol} opened for {ticker} (sell) successfully. Ticket:{order_num}')
                    status_dict[ticker]['curr_pos']=1
                    status_dict[ticker]['ticket_nums'].append(order_num)
                else:
                    write_lines.append(f'{symbol} opened for {ticker} (buy) successfully. ticket: {order_num}')
                    status_dict[ticker]['curr_pos']=0
                    status_dict[ticker]['ticket_nums'].append(order_num)
    return write_lines

def trade():
    global in_trade
    global status_dict
    in_trade=True

    # minimum trade size: 0.01 lots (all quantities rounded to 0.01)
    # tick size: 0.00001 (all prices rounded to 0.00001)
    # IDEAS: might want to consider the benefits of rounding down instead of rounding off (0 likelihood of exceeding margin requirements)
    # write_lines records the logs to be output in a file (by month) after order execution
    write_lines=[f'{dt.datetime.now()}']
    time_start=time.time()
    # bars dict will be populated with bars data through each iteration. Since there is only a possibility of reusing bars only for symbols with 'USD' (for my 'mock' tickers), this bars_dict only saves the bars for these symbols.
    bars_dict={}
    for ticker in trade_tickers:
        ticker_dict=trade_tickers_dict[ticker]
        p_in=ticker_dict['p_in']
        p_out=ticker_dict['p_out']
        long_in=ticker_dict['long_in']
        long_out=ticker_dict['long_out']
        short_in=ticker_dict['short_in']
        short_out=ticker_dict['short_out']
        expected_trade_returns=ticker_dict['param_mean_mean']
        ticker_type=ticker_dict['type']
        if ticker_type=='mock':
            symbol_pairs=ticker_dict['pairs'].split(',')
            symbol_pair_types=ticker_dict['pair_types'].split(',')
        else:
            symbol_lot=ticker_dict['symbol_lot']
            ticker_invert=ticker_dict['invert']
        curr_pos=status_dict[ticker]['curr_pos']
        
        #CALCULATE ADF_PVAL AND ZSCORE==============================
        if ticker_type=='real':
            # Here ticker and symbol are used interchangeably
            # I check if I have polled for the bars before. If I haven't, I'll poll for them, and add them to the dictionary only if the ticker contains 'USD'.
            if ticker not in bars_dict.keys():
                bars=mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_M5, 0, 2880)
                if bars is None:
                    write_lines.append(f'{ticker} error obtaining bars. Error code {mt5.last_error()}')
                    continue
                close_bars=pd.DataFrame(bars)['close']
                if 'USD' in ticker:
                    bars_dict[ticker]=close_bars
            else:
                close_bars=bars_dict[ticker]
            # I am converting a series object to a np array since I don't need the index for adfuller or zscore tests.
            close_bars=close_bars.values
            adf_pval=adfuller(close_bars)[1]
            z_score=ss.zscore(close_bars)[-1]
        else:
            bars_list=[]
            for symbol in symbol_pairs:
                if symbol in bars_dict.keys():
                    bars_list.append(bars_dict[symbol])
                else:
                    bars=mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 2880)
                    if bars is None:
                        write_lines.append(f'{symbol} error obtaining bars. Error code {mt5.last_error()}')
                        break
                    else:
                        close_bars=pd.DataFrame(bars)['close']
                        bars_dict[symbol]=close_bars
                        bars_list.append(close_bars)
            if len(bars_list)!=2:
                continue
            # exr of different 'mock' pairs are calculated based on their pair_types. E.g. exr of RUBXAG is calculated by 1/(USDRUB*XAGUSD). 
            # Because different symbols sometimes have different missing data / no trades, using pandas series for product / division ensures that data from both symbols are aligned in time.
            if symbol_pair_types==['l','s']:
                close_bars=(bars_list[0]/bars_list[1]).dropna().values
            elif symbol_pair_types==['l','l']:
                close_bars=(bars_list[0]*bars_list[1]).dropna().values
            elif symbol_pair_types==['s','l']:
                close_bars=(bars_list[1]/bars_list[0]).dropna().values
            elif symbol_pair_types==['s','s']:
                close_bars=1/(bars_list[1]*bars_list[0]).dropna().values
            # after dropping the unsynced data, I tentatively allow a loss of ~<10% data for the calculation of adf values and z_scores.
            # FUTURE IDEAS: poll more data?
            if len(close_bars)>=2600:
                adf_pval=adfuller(close_bars)[1]
                z_score=ss.zscore(close_bars)[-1]
            else:
                write_lines.append(f'{ticker} not enough data: {len(close_bars.index)}')
                continue
        #SET PAUSE VARIABLES=========================================
        # My additional conditions for opening a long/short position after a stop loss on a long/short position is triggered are:
        # - Time-series must be 'non-stationary' (adf p-value > p_out) OR z_score exceeds the out threshold for the long/short position
        if adf_pval>p_out:
            status_dict[ticker]['lpause']=False
            status_dict[ticker]['spause']=False
        else:
            if z_score>=long_out:
                status_dict[ticker]['lpause']=False
            elif z_score<=short_out:
                status_dict[ticker]['spause']=False
        #TRADE===============================================
        # TRADE LOGIC:
        # if ticker is currently in an open position:
        #   if price series is no longer stationary:
        #       close current position
        #   elif zscore >= long_out:
        #       close current position
        #       if zscore >= short_in:
        #           open short position
        # eliif ticker is currently in a short position:
        #   if price series is no longer stationary:
        #       close current position
        #   elif zscore <= short_out:
        #       close current position
        #       if zscore <= long_in:
        #           open long position
        # elif ticker is not in a position:
        #   if zscore <= long_in:
        #       open long position
        #   elif zscore >= short_in:
        #       open short position
        #
        # CONDITION FOR OPENING A POSITION:
        # Since I am only placing market orders, spread will certainly eat into my profits. Thus for a trade to be profitable its returns should at least cover cost of spread and commision.
        # Returns are calculated as a fraction of principal. Cost per transaction expressed as a fraction of principal = (ask-bid+commission)/(ask or bid)*leverage.
        # I estimate the spread at closing to be = spread at opening a position, and expected returns of my trade = mean returns averaged over the entire 6D parameter space.
        # Condition for opening a position: 2*cost fraction < expected returns.
        # NOTE: commission is left out by accident in the following implementation.

        if curr_pos==0:
            if adf_pval>p_out:
                write_lines.append(f'{ticker} trigger close (sell) due to adf-pval ({adf_pval}) > crit val ({p_out})')
                # if ticker_type=='real --> len(positions)==1. else: len(positions)==2
                write_lines+=close_position(ticker,'ca')
            elif z_score>=long_out:
                write_lines.append(f'{ticker} trigger close (sell) due to z-score ({z_score}) >= long_out ({long_out})')
                write_lines+=close_position(ticker,'cz')
                if z_score>=short_in and status_dict[ticker]['spause']==False:
                    write_lines.append(f'{ticker} trigger open (sell) due to z-score ({z_score}) >= short_in ({short_in})')
                    if ticker_type=='real':
                        write_lines+=open_real(mt5.ORDER_TYPE_SELL,ticker,symbol_lot,ticker_invert,expected_trade_returns)
                    else:
                        write_lines+=open_mock(mt5.ORDER_TYPE_SELL,ticker,symbol_pairs,symbol_pair_types,expected_trade_returns)
        elif curr_pos==1:
            if adf_pval>p_out:
                write_lines.append(f'{ticker} trigger close (buy) due to adf-pval ({adf_pval}) > crit val ({p_out})')
                write_lines+=close_position(ticker,'ca')
            elif z_score<=short_out:
                write_lines.append(f'{ticker} trigger close (buy) due to z-score ({z_score}) <= short_out ({short_out})')
                write_lines+=close_position(ticker,'cz')                    
                if z_score<=long_in and status_dict[ticker]['lpause']==False:
                    write_lines.append(f'{ticker} trigger open (buy) due to z-score ({z_score}) <= long_in ({long_in})')
                    if ticker_type=='real':
                        write_lines+=open_real(mt5.ORDER_TYPE_BUY,ticker,symbol_lot,ticker_invert,expected_trade_returns)
                    else:
                        write_lines+=open_mock(mt5.ORDER_TYPE_BUY,ticker,symbol_pairs,symbol_pair_types,expected_trade_returns)
        elif adf_pval<p_in:
            if z_score>=short_in and status_dict[ticker]['spause']==False:
                write_lines.append(f'{ticker} trigger open (sell) due to z-score ({z_score}) >= short_in ({short_in})')
                if ticker_type=='real':
                    write_lines+=open_real(mt5.ORDER_TYPE_SELL,ticker,symbol_lot,ticker_invert,expected_trade_returns)
                else:
                    write_lines+=open_mock(mt5.ORDER_TYPE_SELL,ticker,symbol_pairs,symbol_pair_types,expected_trade_returns)
            elif z_score<=long_in and status_dict[ticker]['lpause']==False:
                write_lines.append(f'{ticker} trigger open (buy) due to z-score ({z_score}) <= long_in ({long_in})')
                if ticker_type=='real':
                    write_lines+=open_real(mt5.ORDER_TYPE_BUY,ticker,symbol_lot,ticker_invert,expected_trade_returns)
                else:
                    write_lines+=open_mock(mt5.ORDER_TYPE_BUY,ticker,symbol_pairs,symbol_pair_types,expected_trade_returns)
    in_trade=False
    write_lines.append(f'buy-sell order execution time (seconds): {time.time()-time_start}')
    log_f=open('M5_pairs_'+dt.datetime.today().strftime("%Y-%m")+'.txt','a')
    for line in write_lines:
        log_f.write(line+'\n')
        print(line)
    log_f.write('========================================')
    print('========================================')
    log_f.close()
    return

def check_sl():
    # This checks if the loss on each ticker has exceeded 100% (my set limit) every 10 seconds (interval set using the apscheduler cron trigger).
    # This is necessary because mt5 only allows stop orders to be placed for single symbols, and in the case of mock tickers, I am far more concerned if the sum of losses exceed 100% rather than if individual symbols exceed this limit.
    
    # Runs only if trade() is not accessing the status_dict
    global in_trade
    global status_dict
    if in_trade==False:
        write_lines=[]
        acct_info=mt5.account_info()
        if acct_info is None:
            write_lines.append(f'error retrieving account info for trade. Error code: {mt5.last_error()}')
            return
        acct_equity=acct_info.equity
        for ticker in status_dict.keys():
            curr_pos=status_dict[ticker]['curr_pos']
            # curr_pos is the current position of the ticker. I follow my broker's/mt5's convention: 0->long     1->short    2(self-defined)->none
            if curr_pos!=2:
                positions=[mt5.positions_get(ticket=ticket_num)[0] for ticket_num in status_dict[ticker]['ticket_nums']]
                if None in positions:
                    write_lines.append(f'{ticker} failed to obtain position(s). Error code: {mt5.last_error()}')
                    continue
                status_dict[ticker]['positions']=positions
                loss=sum([tradeposition.profit for tradeposition in positions])
                threshloss=0-max_size_proportion*acct_equity
                if loss<threshloss:
                    write_lines.append(f'{ticker} trigger SL close. Loss: {loss}; Max allowed loss: {threshloss}')
                    for tradeposition in positions:
                        if tradeposition.type==1:
                            order_type=mt5.ORDER_TYPE_BUY
                        else:
                            order_type=mt5.ORDER_TYPE_SELL
                        close_symbol=tradeposition.symbol
                        close_ticket=tradeposition.ticket
                        request={
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": close_symbol,
                            "volume": tradeposition.volume,
                            "type": order_type,
                            "position": close_ticket,
                            "comment": f"sl {close_ticket}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                        order=mt5.order_send(request)
                        if order.retcode!=10009 and order.retcode!=10008:
                            write_lines.append(f'{close_symbol} failed to close due to error: {order.retcode} {order.comment}')
                            continue
                        else:
                            write_lines.append(f'{close_symbol} closed successfully')
                    if curr_pos==0:
                        status_dict[ticker]['lpause']=True
                    else:
                        status_dict[ticker]['spause']=True
                    #after closing all positions involved with the ticker, re-initialize status_dict
                    status_dict[ticker]['curr_pos']=2
                    status_dict[ticker]['ticket_nums']=[]
                    status_dict[ticker]['positions']=[]
        # This function runs at a high frequency. I want to log only useful information / error messages
        if write_lines!=[]:
            log_f=open('M5_pairs_'+dt.datetime.today().strftime("%Y-%m")+'.txt','a')
            for line in write_lines:
                log_f.write(line+'\n')
                print(line)
            log_f.write('========================================')
            print('========================================')
            log_f.close()
    for ticker in status_dict.keys():
        status_dict[ticker]['ticket_nums_output']=','.join(str(ticket_num) for ticket_num in status_dict[ticker]['ticket_nums'])
    df=pd.DataFrame.from_dict(status_dict,'index').drop(columns=['positions','ticket_nums']).rename(columns={'ticket_nums_output':'ticket_nums'})
    df.to_csv(curr_status_filedir)
    return   

def EOW_close():
    # By 4pm Fri SGT I want to close all open positions in case they run till end of week when spread is too high.
    # FUTURE WORK: There is a possibility of closing positions with existing opposite ones to reduce cost of spread. Right now this is not very important because the chance of concurrent open tickers is not high.
    global status_dict
    write_lines=[f'{dt.datetime.now()}']
    write_lines.append('EOW closing all positions now')
    all_positions=mt5.positions_get()
    if all_positions is None:
        write_lines.append(f'Error obtaining positions. Error code: {mt5.last_error()}')
        return
    elif len(all_positions)==0:
        write_lines.append('No positions to close')
    else:
        for position in all_positions:
            if position.type==1:
                close_order_type=mt5.ORDER_TYPE_BUY
                close_type='buy'
            else:
                close_order_type=mt5.ORDER_TYPE_SELL
                close_type='sell'
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_order_type,
                "position": position.ticket,
                "comment": f"EOW",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                }
            order=mt5.order_send(request)
            if order.retcode!=10009 and order.retcode!=10008:
                write_lines.append(position.symbol+f' failed to close ({close_type}) due to error: '+order.retcode+' '+order.comment)
            else:
                write_lines.append(position.symbol+' closed ({close_type}) successfully')
    log_f=open('M5_pairs_'+dt.datetime.today().strftime("%Y-%m")+'.txt','a')
    for line in write_lines:
        print(line)
        log_f.write(line+'\n')
    log_f.write('========================================')
    print('========================================')
    log_f.close()
    status_dict={}
    for ticker in trade_tickers:
        status_dict[ticker]={'lpause':False,'spause':False,'curr_pos':2,'ticket_nums':[],'positions':[],'ticket_nums_output':''}
    df=pd.DataFrame.from_dict(status_dict,'index').drop(columns=['positions','ticket_nums']).rename(columns={'ticket_nums_output':'ticket_nums'})
    df.to_csv(curr_status_filedir)
    return
# trade()
# close()
# check_sl()
# FOREX opens on 6 am SGT and closes 6 am Sat (changes due to DST). To avoid holding overnight, close all positions by 4pm Fri.

trade_trigger=OrTrigger([CronTrigger(day_of_week='mon',hour='6-23',minute='*/5',timezone='Asia/Taipei'),
                        CronTrigger(day_of_week='tue-thu',minute='*/5',timezone='Asia/Taipei'),
                        CronTrigger(day_of_week='fri',hour='0-15',minute='*/5',timezone='Asia/Taipei')])
slcheck_trigger=OrTrigger([CronTrigger(day_of_week='mon',hour='6-23',second='*/10',timezone='Asia/Taipei'),
                        CronTrigger(day_of_week='tue-thu',second='*/10',timezone='Asia/Taipei'),
                        CronTrigger(day_of_week='fri',hour='0-15',second='*/10',timezone='Asia/Taipei')])
close_trigger=CronTrigger(day_of_week='fri',hour='16',minute='0',timezone='Asia/Taipei')
scheduler = BlockingScheduler()
trade_job=scheduler.add_job(trade,trade_trigger,misfire_grace_time=10)
slcheck_job=scheduler.add_job(check_sl,slcheck_trigger,misfire_grace_time=2)
close_job=scheduler.add_job(EOW_close,close_trigger,misfire_grace_time=None)
scheduler.start()

while True:
    time.sleep(1)
