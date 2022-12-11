import os
import numpy as np
import math
import csv
import pandas as pd
from itertools import chain

def load_data(path):
    file_list = os.listdir(path)
    dfs = []

    for file in sorted(file_list):
        if file.count(".") == 1:
            name = file.split('.')[0]
        if name in ['ACC', 'EDA', 'IBI']:
            df = pd.read_csv(path + name + '.csv')

            timestamp = float(df.columns[0])
            if name == 'ACC':
                new_df = df.loc[1:,:]
                new_df = new_df.reset_index(drop=True)
                new_df.columns = ['x','y','z']
                timestamp_list = list(map(lambda x,y: x+y*(1/32), [timestamp]*len(new_df), list(new_df.index.values)))
                new_df['timestamp'] = timestamp_list
                new_df = new_df[['timestamp','x','y','z']]
                
            elif name == 'EDA':
                new_df = df.loc[1:,:]
                new_df = new_df.reset_index(drop=True)
                new_df.columns = ['EDA']
                timestamp_list = list(map(lambda x,y: x+y*(1/4), [timestamp]*len(new_df), list(new_df.index.values)))
                new_df['timestamp'] = timestamp_list
                new_df = new_df[['timestamp', 'EDA']]
                

            elif name == 'IBI':
                df.columns = ['timestamp', 'IBI']
                df['timestamp'] = df['timestamp'] + timestamp
                df['IBI'] = df['IBI'] * 1000
                new_df = df.copy()
                
            new_df = new_df.reset_index(drop=True)   
            dfs.append(new_df)
    return dfs


path1 = f'./preprocess/1/'
path2 = f'./preprocess/2/'
# path3 = f'./preprocess/3/'

dfs1 = load_data(path1)
dfs2 = load_data(path2)
# dfs3 = load_data(path3)

dfs1[0] = pd.concat([dfs1[0], dfs2[0]])
dfs1[1] = pd.concat([dfs1[1], dfs2[1]])
dfs1[2] = pd.concat([dfs1[2], dfs2[2]])

dfs1[0].to_csv(path1 + 'ACC.csv')
dfs1[1].to_csv(path1 + 'EDA.csv')
dfs1[2].to_csv(path1 + 'IBI.csv')