from GCP_Option_Hedging_Toolkit import *
import xmlrpc.client
'''Strategy is to hold the position for a certain number of days. In case the exit day falls on a weekend,
 function will return the cloest on which position will be disposed.'''
def set_params():
    '''Parameters to play with for different trading/hedging thresholds'''
    global stratcode,expcode,trade_gen_func,trading_entry_threshold,contract_expiry_days,trade_close_period,raw_data_file,atm_vol_file,foreign_notional,bt_start_date,bt_end_date,volsprd,undsprd,delta_hedging_threshold_long,delta_hedging_threshold_short,logfolder,log_path,vol_surface_server_add
    stratcode=sys.argv[1]
    expcode = sys.argv[2]
    trade_gen_func = sys.argv[3]
    trading_entry_threshold = float(sys.argv[4])
    contract_expiry_days = float(sys.argv[5])
    trade_close_period = float(sys.argv[6])
    raw_data_file = sys.argv[7]
    atm_vol_file = sys.argv[8]
    foreign_notional = float(sys.argv[9])
    volsprd = float(sys.argv[10])
    undsprd = float(sys.argv[11])
    bt_start_date = dt.datetime.strptime(sys.argv[12], '%Y-%m-%d')
    bt_end_date = dt.datetime.strptime(sys.argv[13], '%Y-%m-%d')
    delta_hedging_threshold_long = float(sys.argv[14])
    delta_hedging_threshold_short = float(sys.argv[15])
    logfolder = sys.argv[16]
    log_path = logfolder + stratcode+"\\"+expcode + "\\"
    os.system("mkdir " + log_path)
    vol_surface_server_add=sys.argv[17]

def closet_weekday(date_list):
    weekday = []
    for n in range(len(date_list)):
        if date_list[n].weekday()>4:
            offset = 1 if date_list[n].weekday() == 5 else 2
            date_ = date_list[n] - dt.timedelta(days=offset)
            weekday.append(date_)
        else:
            weekday.append(date_list[n])
    return weekday

#delta for call:[0,1] put:[-1,0]
'''Function to look up the implied volatility matrix for the implied volatility on that trading day in order to
price the option under the BSM model.'''
def implied_vol_look_up(df,vol_matrix):
    implied_vol = []
    for i in range (len(df)):
        implied_vol.append(vol_matrix.loc[df.index[i]][df.day_diff[i]])
    return implied_vol

def implied_vol_look_up_diff(df,vol_matrix):
    implied_vol = []
    for i in range(len(df)):
        if((i+1) < len(df)):
            implied_vol.append(vol_matrix.loc[df.index[i+1]][df.day_diff[i]]-vol_matrix.loc[df.index[i]][df.day_diff[i]])
        else:
            implied_vol.append(0)
    return implied_vol

def trade_gen(df_main,trade_gen_func,trading_entry_threshold):
    if trade_gen_func == "voldiff":
        df_main['signal'] =signal_gen_func_001(df_main,trading_entry_threshold)
    if trade_gen_func == "volrollstd":
        df_main['signal'] = signal_gen_func_002(df_main, trading_entry_threshold)
    return df_main

'''Currently logic001: Long options if implied volatility < past 2 M realized volatility. Short options 
    if implied volatility > past 2 M realized volatility.'''
def signal_gen_func_001(df,trading_entry_threshold):
    res=np.where(df.TM_implied_vol - df.TM_realized_vol > trading_entry_threshold, -1,
             np.where(df.TM_implied_vol - df.TM_realized_vol < -trading_entry_threshold, 1, 0))
    return res

def signal_gen_func_002(df,trading_entry_threshold):
    df['vol_spread'] = df.TM_implied_vol - df.TM_realized_vol
    df['Rolling_signal_pos'] = df['vol_spread'].rolling(250).mean() + df['vol_spread'].rolling(250).std() * trading_entry_threshold
    df['Rolling_signal_pos'].fillna(value=0)
    df['Rolling_signal_neg'] = df['vol_spread'].rolling(250).mean() - df['vol_spread'].rolling(250).std() * trading_entry_threshold
    df['Rolling_signal_neg'].fillna(value=0)
    res = np.where(df.Rolling_signal_neg > df.vol_spread, 1,
             np.where(df.Rolling_signal_pos < df.vol_spread, -1, 0))
    plt.figure(figsize=(15, 10))
    plt.grid(True)
    plt.plot(df.vol_spread)
    plt.plot(df.Rolling_signal_pos)
    plt.plot(df.Rolling_signal_neg)
    plt.xticks(rotation=45)
    plt.title('Vol Spread (Entry threshold = ' + str(round(trading_entry_threshold, 2)) + ' vol')
    plt.savefig(log_path+"volrollstd_"+expcode+".png")
    return res
def get_vol_by_params(date,tenor,delta):
    if delta>=0:
        x = 1-delta
    else:
        x=abs(delta)
    return (vol_surface[date][tenor](x)).tolist()
#assumptions: spot move-delta move-vol move- px and greeks move
def simulate_delta_and_vol(df,exp):
    call_delta = []
    put_delta = []
    implied_vol_call=[]
    implied_vol_put = []
    for n in range(len(df)):
        if n==0:
            implied_vol_call.append(get_vol_by_params(df.index[n],df.loc[df.index[n]]['day_diff'],0.5))
            implied_vol_put.append(get_vol_by_params(df.index[n], df.loc[df.index[n]]['day_diff'], -0.5))
            call_option = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                             implied_vol_call[n] / 100, init_date=df.index[n], exp_date=exp,
                                             type='call')
            put_option = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                            implied_vol_put[n]/ 100, init_date=df.index[n], exp_date=exp,
                                            type='put')
            call_option.recalculate()
            put_option.recalculate()
            call_delta.append(call_option.delta())
            put_delta.append(put_option.delta())

        else:
            call_option = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                             implied_vol_call[n-1] / 100, init_date=df.index[n], exp_date=exp,
                                             type='call')
            put_option = Prepare_BSM_Option(df.spot_rate[n], strike, df.interest_rate[n],0,
                                            implied_vol_put[n-1] / 100, init_date=df.index[n], exp_date=exp,
                                            type='put')
            call_option.recalculate()
            put_option.recalculate()
            call_delta.append(call_option.delta())
            put_delta.append(put_option.delta())
            implied_vol_call.append(get_vol_by_params(df.index[n], df.loc[df.index[n]]['day_diff'], call_delta[-1]))
            implied_vol_put.append(get_vol_by_params(df.index[n], df.loc[df.index[n]]['day_diff'], put_delta[-1]))

    df['call_delta'] = np.array(call_delta)
    df['put_delta'] = np.array(put_delta)
    df['implied_vol_call']=np.array(implied_vol_call)
    df['implied_vol_put'] = np.array(implied_vol_put)
    df['option_delta'] = df['call_delta']  + df['put_delta']
    return df
   
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    '''Step 0. Loading params from batch file'''
    set_params()

    '''Step 1. Loading necessary data from disk drive in order to perform analysis'''
    #Loading data, convert column names to a more understandable fashion
    columns = ['fwd_premium', 'spot_rate', 'TM_realized_vol', 'TM_implied_vol']
    df_main = pd.read_csv(raw_data_file, header = 0, index_col=[0], parse_dates=[0])
    df_main.columns = columns
    trading_day_list = df_main.index

    #Loading implied volatility matrix.
    df_vol = pd.read_csv(atm_vol_file, header = 0, index_col= [0], parse_dates=[0])
    h_vol = xmlrpc.client.ServerProxy(vol_surface_server_add)
    vol_surface=pickle.loads((h_vol.get_vol_surface()).data)

    '''Step 1. Generate variables for BSM model inputs:  Strike, interest rate, expiry date, position closing date'''
    #calculate fwd price - this price will be our strike price
    df_main['fwd_price'] = df_main.spot_rate + (df_main.fwd_premium/10000)
    #interest rate r will be derived from PPP formula according to the function in the toolkit
    df_main['interest_rate'] = calculate_US_interest_rate(df_main.spot_rate, df_main.fwd_price)
    #expiry date to be 60 days away from the trade initiation date
    df_main['expiry_date'] = df_main.index + dt.timedelta(days=contract_expiry_days)
    df_main.expiry = closet_weekday(df_main['expiry_date'])

    #calculate the trade closing date
    df_main['close_date'] = df_main.index + dt.timedelta(days=trade_close_period)
    df_main.close_date = closet_weekday(df_main['close_date'])

    '''Step 2. Generate trading signals to initiate trades in order to evaluate hedging activities.'''
    df_main = trade_gen(df_main, trade_gen_func, trading_entry_threshold)

    '''Collect samples where there is a trade'''
    back_test_sample = df_main.loc[df_main.expiry_date <= df_main.index[-1]]
    back_test_sample = df_main.loc[(df_main.index>=bt_start_date) & (df_main.index<=bt_end_date)]
    bt_day_list=back_test_sample.index
    back_test_sample = back_test_sample.loc[back_test_sample.signal != 0]

    '''Step 3. Looping over the collected samples. Calculate the PNL outcome and the detailed trade log of each trade'''
    PNL_distribution = []
    gamma_pnl=[]
    delta_pnl_hedged=[]
    delta_pnl_unhedged = []
    theta_pnl=[]
    spot_cost=[]
    arar=[]   #absolute risk adjusted return
    sub_df_dict = {}
    sub_columns = ['spot_rate', 'fwd_price', 'interest_rate', 'signal']
    Descriptive = []
    for i in range(len(back_test_sample)):

        #determine variables: expiry, strike, closing date
        test_date = back_test_sample.index[i]
        close_date = back_test_sample.close_date[i]
        exp = back_test_sample.expiry_date[i]
        strike = back_test_sample.fwd_price[i]

        #cut the next 30 days of data out from the main dataframe for simulation of the trade
        sub_df = df_main.loc[(df_main.index >= test_date) & (df_main.index <= close_date)]
        print('simulating results for trade entered on EOD of : ' + test_date.date().strftime("%Y-%m-%d"))
        # not all the columns of the main data frame is useful, only select those that are relevant to hedging
        sub_df = sub_df[sub_columns]
        sub_df['day_diff'] = (exp - sub_df.index).days

        #sub_df['implied_vol'] = implied_vol_look_up(sub_df, df_vol)
        sub_df['implied_vol_diff'] = implied_vol_look_up_diff(sub_df, df_vol)

        # simulate according to the toolkit functions
        sub_df = simulate_delta_and_vol(sub_df,exp)
        sub_df = BSM_Option_Pricing(sub_df,strike,exp,volsprd)
        sub_df['trade_signal'] = sub_df.signal[0]
        sub_df['foreign_expo'] = foreign_notional
        sub_df['option_delta'] = (sub_df.option_delta) * sub_df.trade_signal
        sub_df = hedging_strategy_simulation(sub_df, sub_df.signal[0],delta_hedging_threshold_long,delta_hedging_threshold_short)
        sub_df = calculate_PNL(sub_df,undsprd)
        sub_df = pnlByGreeks(sub_df)

        #collect the PNL of this trade into the list, collect the trade data from into the dictionary
        PNL_distribution.append(sub_df.PNL.sum())
        gamma_pnl.append(sub_df.option_Gamma_PnL.sum())
        delta_pnl_hedged.append(sub_df.option_hedged_delta_PnL.sum())
        delta_pnl_unhedged.append(sub_df.option_unhedged_delta_PnL.sum())
        theta_pnl.append(sub_df.option_Theta_PnL.sum())
        spot_cost.append(sub_df.und_cost.sum())
        arar.append(np.nan_to_num((252*sub_df.PNL.mean())/(np.sqrt(252)*sub_df.PNL.std())))
        sub_df_dict[test_date] = sub_df

    '''Part 4. Plot the PNL distribution of the PNL.'''
    Descriptive.append([delta_hedging_threshold_long,delta_hedging_threshold_short,np.mean(PNL_distribution), np.std(PNL_distribution), skew(PNL_distribution), kurtosis(PNL_distribution),
                        np.mean(PNL_distribution)/ np.std(PNL_distribution)])
    plt.figure(figsize=(15, 15))
    plt.figtext(0.7, 0.8, "Mean: %.2f" % Descriptive[0][2])
    plt.figtext(0.7, 0.7, "STD: %.2f" % Descriptive[0][3])
    plt.figtext(0.7, 0.6, "Skewness: %.2f" % Descriptive[0][4])
    plt.figtext(0.7, 0.5, "Kurtosis: %.2f" % Descriptive[0][5])
    plt.figtext(0.7, 0.4, "Risk Adj.: %.2f" % Descriptive[0][6])
    plt.rcParams.update({'font.size': 22})
    plt.grid(True)
    plt.title('Sub strategy PNL Distribution - experiment code='+expcode)
    plt.xlabel('in USD million')
    plt.hist(PNL_distribution, bins=100)
    plt.savefig(log_path + 'histo_'+expcode+'.png')

    '''Part 5. PNL curve from start date to close date and plot'''
    agg_daily_pnl = []

    for i in range(len(bt_day_list)):
        temp_sum = 0
        print("Calculating daily pnl for trading day " + bt_day_list[i].date().strftime("%Y-%m-%d"))
        look_up_range_begin_dt=bt_day_list[i]-dt.timedelta(days=trade_close_period+1)
        look_up_range=[t for t in trading_day_list if t>=look_up_range_begin_dt and t<=bt_day_list[i]]
        for j in range(len(look_up_range)):
            try:
                temp_sum += np.nan_to_num(sub_df_dict[look_up_range[j]].loc[bt_day_list[i]]['PNL'])
            except:
                temp_sum += 0
        agg_daily_pnl.append(temp_sum)

    agg_pnl_df = pd.DataFrame.from_dict({'date':bt_day_list,'daily_pnl':agg_daily_pnl})
    agg_pnl_df['cum_pnl']=agg_pnl_df.daily_pnl.cumsum()
    agg_pnl_df['histmax']=agg_pnl_df['cum_pnl'].rolling(len(bt_day_list), min_periods=1).max()
    agg_pnl_df['drawdown']=agg_pnl_df['cum_pnl']-agg_pnl_df['histmax']

    plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 15})
    plt.plot(bt_day_list,agg_pnl_df.cum_pnl)
    plt.grid(True)
    plt.title('Aggregate PNL Curve by dates - experiment code='+expcode)
    plt.xlabel('in USD million')
    plt.savefig(log_path + 'linechart_'+expcode+'.png')

    '''Part 6. Saving files and charts'''
    back_test_sample.to_csv(log_path+"tradebook_"+expcode+".csv")
    res_dict={
              'stratcode':[stratcode],
              'expcode':[expcode],
              'trade_gen_func':[trade_gen_func],
              'trading_entry_threshold':[trading_entry_threshold],
              'contract_expiry_days':[contract_expiry_days],
              'trade_close_period':[trade_close_period],
              'bt_start_date':[bt_start_date],
              'bt_end_date':[bt_end_date],
              'foreign_notional':[foreign_notional],
              'volsprd':[volsprd],
              'undsprd':[undsprd],
              'delta_hedging_threshold_long':[delta_hedging_threshold_long],
              'delta_hedging_threshold_short':[delta_hedging_threshold_short],

              'no_bt_day':[len(bt_day_list)],
              'no_trades': [len(back_test_sample)],
              'bt_time_span':[float((bt_day_list[-1]-bt_day_list[0]).days)/365], #timespan in years
              'max_drawdown': [agg_pnl_df['drawdown'].min()],
              'avg_daily_pnl': [agg_pnl_df.daily_pnl.mean()],
              'daily_pnl_std': [agg_pnl_df.daily_pnl.std()],
              'risk_adjust_return': [np.nan_to_num(agg_pnl_df.daily_pnl.mean()/agg_pnl_df.daily_pnl.std())],

              'option_avg_gamma_pnl':[np.mean(gamma_pnl)],
              'option_avg_delta_pnl_hedge':[np.mean(delta_pnl_hedged)],
              'option_avg_delta_pnl_unhedge': [np.mean(delta_pnl_unhedged)],
              'option_avg_theta_pnl':[np.mean(theta_pnl)],
              'option_avg_spot_cost':[np.mean(spot_cost)],
              'option_avg_arar':[np.mean(arar)],
              'option_avg_pnl':[np.mean(PNL_distribution)],
              'option_pnls_dev':[np.std(PNL_distribution)],
              'pnl_skewness':[Descriptive[0][4]],
              'pnl_kutosis':[Descriptive[0][5]]
              }
    res_df=pd.DataFrame.from_dict(res_dict)
    res_df.to_csv(log_path+"res.csv", index=False)
    print("Result saved to "+log_path+"res.csv")
