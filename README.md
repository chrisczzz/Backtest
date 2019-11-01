# Backtest
A simple backtest system to how long/short volatility with hedging performs for EUR/USD asset in the past 3 years.


3 main files:
  - Option Hedging Server: Load implied volatility and related data and construct and store multiple volatility surfaces under
    port at localhost;
    
  - Option Hedging Toolkit: Various pricing functions for back testing purposes. Eg. BSM option pricing, option greeks calculations.
  - Option Hedging Backtest: Main file that simulates trading strategies and provides related metrics and graphs. Eg. Cumulative PNL, PNL distribution and so on.
