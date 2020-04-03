#%%
import os
import pandas as pd
import tqdm
top_cryptos = [
 'ETH-BTC.parquet'
,'XRP-BTC.parquet'
,'BNB-BTC.parquet'
,'XLM-BTC.parquet'
,'LTC-BTC.parquet'
,'TUSD-BTC.parquet'
,'USDT-BTC.parquet'
]


dirname = "D:\\Binance\\binance-full-history"
os.chdir(dirname)
os.makedirs('csv_files', exist_ok=True)
for filename in  os.listdir(dirname):
    if  filename in top_cryptos:
        full_path = f'{dirname}\\{filename}'
        df = pd.read_parquet(full_path)

        new_filename = filename.replace('.parquet', '.csv')
        new_full_path = f'csv_files/{new_filename}'
        df.to_csv(new_full_path)
        





# %%
