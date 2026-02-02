# Futures_Data_Analysis
This is a tool i made to analyse as many CSV files containing futures data you want to, this will give you contract wise data containing metrics related to volume/open interest and details about loss/gain making days aswell as yearly analysis of a stock/index.
you have to create a folder called "data_folder" and upload as many CSV files with historical futures data from NSE or you can modify this to analyze data from any exchange.
make sure your "data_folder" is structured in this manner:
  your_project/
├── data_folder/              # Place all your CSV files here
│   ├── FUTIDX_BANKNIFTY_...csv
│   ├── FUTSTK_TATAMOTORS_...csv
│   └── ... (add as many CSV files as needed)
├── stock_analysis.py # This is the Source Code
