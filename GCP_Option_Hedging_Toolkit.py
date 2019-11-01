import numpy as np
import QuantLib as qt
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import warnings
import sys
import logging
import os
import scipy as sci
import pickle
'''
   Package: GCP Project Option Hedging
   Script: GCP Option Hedging Toolkit
   
   Overview: Script acts as a backend support function tool list which collects a series of common functions 
   by GCP Back test script and GCP Monte Carlo simulations.
   
   Given script's supportive nature, an overview of script process is omitted. Brief descriptions of each functions
   is given below.
'''


'''Function to calculate interest rate for one currency of the pair given a fixed interest rate of the other. Method based
   on formula under Purchasing Power Parity (PPP). 
   Parameters:  spot_price: current spot exchange rate of the currency pair
                fwd_price: the forward price of the currency pair.
                EUR_rate: default 0, the interest rate of home currency
                forward period: default 60, the period over which the fwd price is quoted.
  '''
def calculate_US_interest_rate(spot_price, fwd_price, EUR_rate = 0, forward_period = 60):
    us_rate = ((fwd_price/ spot_price) * (EUR_rate + 1)) - 1
    us_rate = (us_rate/forward_period)*360

    return us_rate


'''Function to derive forward price of a currency pair.  Method based on formula under Purchasing Power Parity (PPP). 
   Parameters:  spot_price: current spot exchange rate of the currency pair
                us_rate: the interest rate of foreign currency
                EUR_rate: default 0, the interest rate of home currency
                forward period: default 60, the period over which the fwd price is quoted.
  '''
def calculate_forward_price(spot_price,us_rate, EUR_rate = 0, forward_period = 60):
    fwd_price = spot_price * ((1+us_rate * (forward_period/360))/(1+EUR_rate *(forward_period/360)))
    return fwd_price


'''Wrapper function to create a BSM class object to price options through the BSM model.
   Given Quantlib's OOP implementation method - initialize an Quantlib OOP object whcih requires the following
   information during initialization. Initialization method follows Quantlib tutorials.
        
        Spot: Current underlying price
        Strike: Option strike price
        Volatility: underlying asset volatility
        init_date: date when option is analyzed
        exp_date: option expiry date
        type: option type
'''
def Prepare_BSM_Option(spot, strike, rd, rf, vol,init_date,exp_date, type = 'call'):
    today = qt.Date(init_date.day, init_date.month, init_date.year)
    qt.Settings.instance().evaluationDate = today
    if type == 'call':
        option = qt.EuropeanOption(qt.PlainVanillaPayoff(qt.Option.Call, strike),
                                qt.EuropeanExercise(qt.Date(exp_date.day, exp_date.month, exp_date.year)))
    else:
        option = qt.EuropeanOption(qt.PlainVanillaPayoff(qt.Option.Put, strike),
                                qt.EuropeanExercise(qt.Date(exp_date.day, exp_date.month, exp_date.year)))

    u = qt.SimpleQuote(spot)
    rd = qt.SimpleQuote(rd)
    rf = qt.SimpleQuote(rf)
    sigma = qt.SimpleQuote(vol)
    rf = qt.FlatForward(0, qt.TARGET(), qt.QuoteHandle(rf), qt.Actual360())
    rd = qt.FlatForward(0, qt.TARGET(), qt.QuoteHandle(rd), qt.Actual360())
    volatility = qt.BlackConstantVol(0, qt.TARGET(), qt.QuoteHandle(sigma), qt.Actual360())
    process = qt.GarmanKohlagenProcess(qt.QuoteHandle(u), qt.YieldTermStructureHandle(rf),
                                       qt.YieldTermStructureHandle(rd), qt.BlackVolTermStructureHandle(volatility))
    engine = qt.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    return option
'''Function to return a dataframe with straddle price. Given Quantlib's OOP implementation method, which does not
    support array operation, function deploys method that goes through each line of given data frame in order to
    calculate option prices. Finally a list of option prices are appended to the data frame with column 'straddle',
    and the associated delta under column 'option delta'''
def BSM_Option_Pricing(df, strike, exp,volsprd):
    call_price = []
    call_gamma = []
    call_vega = []
    call_rho = []
    call_theta = []
    put_price = []
    put_gamma = []
    put_vega = []
    put_rho = []
    put_theta = []
    call_price_bid=[]
    call_price_ask=[]
    put_price_bid=[]
    put_price_ask=[]
    for n in range(len(df)):
        call_option = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                         df.implied_vol_call[n] / 100, init_date=df.index[n], exp_date=exp,
                                         type='call')
        call_option_bid=Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                           (df.implied_vol_call[n]-0.5*volsprd) / 100, init_date=df.index[n], exp_date=exp,
                                         type='call')
        call_option_ask = Prepare_BSM_Option(df.spot_rate[n], strike,df.interest_rate[n],0,
                                             (df.implied_vol_call[n] + 0.5 * volsprd) / 100, init_date=df.index[n],
                                             exp_date=exp,
                                             type='call')

        put_option = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                        df.implied_vol_put[n] / 100, init_date=df.index[n], exp_date=exp,
                                        type='put')
        put_option_bid = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                            (df.implied_vol_put[n]-0.5*volsprd) / 100, init_date=df.index[n], exp_date=exp,
                                        type='put')
        put_option_ask = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                            (df.implied_vol_put[n] + 0.5 * volsprd) / 100, init_date=df.index[n],
                                            exp_date=exp,
                                            type='put')

        call_option.recalculate()
        call_price.append(call_option.NPV())
        call_gamma.append(call_option.gamma())
        call_vega.append(call_option.vega())
        call_rho.append(call_option.rho())
        call_theta.append(call_option.thetaPerDay())
        put_option.recalculate()
        put_price.append(put_option.NPV())
        put_gamma.append(put_option.gamma())
        put_vega.append(put_option.vega()/100)
        put_rho.append(put_option.rho()/100)
        put_theta.append(put_option.thetaPerDay())

        call_option_bid.recalculate()
        call_price_bid.append(call_option_bid.NPV())
        call_option_ask.recalculate()
        call_price_ask.append(call_option_ask.NPV())
        put_option_bid.recalculate()
        put_price_bid.append(put_option_bid.NPV())
        put_option_ask.recalculate()
        put_price_ask.append(put_option_ask.NPV())

    df['straddle'] = np.array(call_price) + np.array(put_price)
    df['straddle_bid']=np.array(call_price_bid) + np.array(put_price_bid)
    df['straddle_ask']=np.array(call_price_ask) + np.array(put_price_ask)
    df['option_gamma'] = (np.array(call_gamma) + np.array(put_gamma))
    df['option_vega'] = (np.array(call_vega) + np.array(put_vega))
    df['option_rho'] = (np.array(call_rho) + np.array(put_rho))
    df['option_theta'] = (np.array(call_theta) + np.array(put_theta))
    return df

'''Function to produce dataframe columns that track the hedging activities and post hedging delta according to
the global variable. Given its dynamic nature, method is implemented through a for loop instead of an array operation.
Overview of logic is as follows:
    1. For every step in time, determine the unhedged amount of delta
    2. If unhedged delta surpasses hedging threshold - generate hedging signal. 
        The amount of delta to hedge is -unhedged delta in order to bring portfolio to delta neutral.
    3. Continue timestep, hedge once again when unhedged delta passes the threshold.
'''
def hedging_strategy_simulation(df,dir,delta_hedging_threshold_long,delta_hedging_threshold_short):
    if dir>=0:
        delta_hedging_threshold=delta_hedging_threshold_long
    else:
        delta_hedging_threshold=delta_hedging_threshold_short
    post_hedging_deltas = [0]
    hedged_delta_list = [0]
    hedged_delta = 0.
    for n in range(1, len(df)):
        unhedged_delta = df['option_delta'][n] + hedged_delta
        if abs(unhedged_delta) < delta_hedging_threshold:
            post_hedging_deltas.append(df['option_delta'][n] + hedged_delta)
            hedged_delta_list.append(0)
        else:
            hedged_delta -= unhedged_delta
            post_hedging_deltas.append(df['option_delta'][n] + hedged_delta)
            hedged_delta_list.append(unhedged_delta)
            unhedged_delta = 0.

    df['post_hedge_delta'] = post_hedging_deltas
    df['hedge_action'] = -1 * np.array(hedged_delta_list)

    return df


'''Function to determine the profit and loss performance.
    P&L = gain or loss due to change of option prices in the USD term
        + gain or loss due to change of currency prices that are longed/shorted during the hedging process'''

def calculate_PNL(df,undsprd):

    df['underlying_purchase'] = df['hedge_action'] * df['foreign_expo']
    df['und_cost']=df['underlying_purchase'].abs()*0.5*undsprd
    df['underlying_position_cumsum'] = df.underlying_purchase.cumsum()
    df['mid_price_change'] = df['spot_rate'] - df['spot_rate'].shift(1)
    df['underlying_PNL_USD'] = df.underlying_position_cumsum.shift(1) * df.mid_price_change
    df['ref_option_price']=df['straddle']
    if df['trade_signal'][0]==1:
        df.set_value(df.index[0], 'ref_option_price', df['straddle_ask'][0])
        df.set_value(df.index[-1], 'ref_option_price', df['straddle_bid'][-1])
    if df['trade_signal'][0]==-1:
        df.set_value(df.index[0], 'ref_option_price', df['straddle_bid'][0])
        df.set_value(df.index[-1], 'ref_option_price', df['straddle_ask'][-1])
    df['option_price_change'] = df['ref_option_price'] - df['ref_option_price'].shift(1)
    df['option_PNL_USD'] = df.trade_signal * df.foreign_expo.shift(1) * df.option_price_change
    df['PNL'] = df.underlying_PNL_USD -df.und_cost+ df.option_PNL_USD
    df['cum_PNL'] = df.PNL.cumsum()

    return df

def pnlByGreeks(df):
    df['option_Gamma_PnL'] = (df.trade_signal * df.foreign_expo * df['option_gamma'] * 0.5 * (
                df['spot_rate'].shift(-1) - df['spot_rate']) ** 2).shift(1)
    df['option_Vega_PnL'] = (
                df.trade_signal * df.foreign_expo * df['option_vega'] * df['implied_vol_diff'] / 100).shift(1)
    df['option_Theta_PnL'] = df.trade_signal * df.foreign_expo * df['option_theta'].shift(1)
    df['option_Rho_PnL'] = (df.trade_signal * df.foreign_expo * df['option_rho'] * (
                df['interest_rate'].shift(-1) - df['interest_rate'])).shift(1)
    df['option_unhedged_delta_PnL'] = (df.trade_signal * df.foreign_expo * df['option_delta']
                                       * (df['spot_rate'].shift(-1) - df['spot_rate'])).shift(1)
    df['option_hedged_delta_PnL'] = ((df.trade_signal * df.foreign_expo * df['option_delta']
                                      * (df['spot_rate'].shift(-1) - df['spot_rate'])) + df[
                                         'underlying_PNL_USD']).shift(1)
    return df
