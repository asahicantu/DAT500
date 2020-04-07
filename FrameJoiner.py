
#%%
import pandas as pd
import sys
import os

#%%
def MergeDataFrames(dir_path,target_file_name):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and 'csv' in  f]
    dfs= []
    for file in files:
        print(file)
        df = pd.read_csv(file,index_col='Date')
        dfs.append(df)
    main_df = pd.concat(dfs,axis=1,sort=False) 
    main_df.to_csv(target_file_name)

def MergeDataFramesVertical(dir_path,pattern,source_extension,index_column):
    files = [(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and pattern in  f and source_extension in f]
    dfs= []
    for file in files:
        file_path = os.path.join(file[0],file[1])
        file_prefix = file[1].replace(f'.{source_extension}','').replace(pattern,'')
        print(file_path)
        df = pd.read_csv(file_path,index_col=index_column)
        df.insert(0, "Currency", file_prefix) 
        dfs.append(df)
    main_df = pd.concat(dfs,axis=0,sort=False) 
    return main_df

def getDataFrames(dir_path):
    dfs = {}
#%%

main_df  = MergeDataFramesVertical('..\\DataSets\\Misc','CoinmarketCap_','csv','Date')
main_df.to_csv('..\\DataSets\\CoinMarketCap.csv')
#%%
main_df  = MergeDataFramesVertical('..\\DataSets','Binance_','csv',None)
main_df.to_csv('..\\DataSets\\Binance.csv')



# %%
