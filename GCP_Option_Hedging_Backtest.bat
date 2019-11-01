set stratcode=strat001
set expcode=exp003
rem options:voldiff,volrollstd
set trade_gen_func=volrollstd
set trading_entry_threshold=1
set contract_expiry_days=60
set trade_close_period=30
set raw_data_file=rawdata_2M.csv
set atm_vol_file=EUR_IMPVOL_ATM.csv
set foreign_notional=50
rem in vol
set volsprd=0.15
rem 1 bp
set undsprd=0.0001
set bt_start_date=2015-01-01
set bt_end_date=2015-03-01
set delta_hedging_threshold_long=0.1
set delta_hedging_threshold_short=0.2
set logfolder=D:\master\courses\GCP\log\
set vol_surface_server_add=http://localhost:8000

python GCP_Option_Hedging_Backtest.py %stratcode% %expcode% %trade_gen_func% %trading_entry_threshold% %contract_expiry_days% %trade_close_period% %raw_data_file% %atm_vol_file% %foreign_notional% %volsprd% %undsprd% %bt_start_date% %bt_end_date% %delta_hedging_threshold_long% %delta_hedging_threshold_short% %logfolder% %vol_surface_server_add%
