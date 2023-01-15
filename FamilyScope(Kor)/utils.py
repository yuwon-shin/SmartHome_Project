import os
import numpy as np
import math
import csv
from itertools import chain
from biosppy.signals import eda
from scipy.stats import median_abs_deviation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.set_option('mode.chained_assignment',  None)


global activities
activities = ['baseline','식사','영상 시청', '보드 게임', '청소']

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

def load_data_ver2(path):
    file_list = os.listdir(path)
    dfs = []

    for file in sorted(file_list):
        if file.count(".") == 1:
            name = file.split('.')[0]
        if name in ['ACC', 'EDA', 'IBI']:
            new_df = pd.read_csv(path + name + '.csv')
            new_df = new_df.reset_index(drop=True)   
            dfs.append(new_df)
    return dfs

def count_peaks(df, wnd):
    edadata = df['EDA'].tolist()
    timestamps = df['timestamp'].tolist() 
    out = eda.kbk_scr(edadata, sampling_rate = 4, min_amplitude=0.01)
    peaks = out['peaks'].tolist()  
    
     
    eda_scr = pd.DataFrame(data={'timestamp': timestamps} )
    eda_scr = eda_scr.reset_index(drop=True)

    new_col = np.zeros(len(df))
    new_col[peaks] = 1
    eda_scr['peak'] = new_col

    peaks = []
    timestamp = []
    STEP = 4*wnd
    for i in range(0,len(eda_scr), STEP):
        df_partial = eda_scr.iloc[i:i+STEP,:]
        plen = len(df_partial)
        if plen < STEP:
            continue
        peak = sum(df_partial['peak'])
        peaks.append(peak)
        timestamp.append(df_partial.iloc[0,0])
    eda_peaks = pd.DataFrame(data={'timestamp':timestamp, 'peak_per_min': peaks})
    eda_peaks['peak_per_min'] = eda_peaks['peak_per_min'].replace(0, np.NaN)
    eda_peaks['peak_per_min'] = eda_peaks['peak_per_min'].interpolate()
    

    return eda_peaks

def calculate_HRV(df,wnd: int):
    
    RR_list = list(np.array(df['IBI']))
    RR_sqdiff = []
    RR_diff_timestamp = []
    cnt = 0
    while (cnt < (len(RR_list)-1)): 
        RR_sqdiff.append(math.pow(RR_list[cnt+1] - RR_list[cnt], 2))

        RR_diff_timestamp.append(df['timestamp'][cnt])
        cnt += 1
    RR_sqdiff_df = pd.DataFrame({'timestamp': RR_diff_timestamp, 'RR_sqdiff': RR_sqdiff})

    RMSSD_timestamp = []
    RMSSD = []
    startT = RR_sqdiff_df['timestamp'].min()
    
    while startT < RR_sqdiff_df['timestamp'].max():
        endT = startT + wnd
        temp = list(np.array(RR_sqdiff_df.loc[(RR_sqdiff_df['timestamp'] >= startT)&(RR_sqdiff_df['timestamp'] <= endT),['RR_sqdiff']]))
        if len(temp) > 1:
            rmssd_data = math.sqrt(np.sum(temp)/(len(temp)-1))
        else:
            rmssd_data = math.sqrt(np.sum(temp))
        RMSSD.append(rmssd_data)
        RMSSD_timestamp.append(startT)
        startT += wnd

    HRV_df = pd.DataFrame({'timestamp': RMSSD_timestamp, 'RMSSD':RMSSD})
    HRV_df['RMSSD'] = HRV_df['RMSSD'].replace(0, np.NaN)
    HRV_df['RMSSD'] = HRV_df['RMSSD'].interpolate(method='values')

    return HRV_df

def calculate_active(acc_df, wnd):
    timestamp = []
    MAG = []
    startT = acc_df['timestamp'].min()
    
    while startT < acc_df['timestamp'].max():
        endT = startT + wnd
        temp = list(np.array(acc_df.loc[(acc_df['timestamp'] >= startT)&(acc_df['timestamp'] <= endT),['mag']]))
        
        dd = np.mean(temp)
        MAG.append(dd)
        timestamp.append(startT)
        startT += wnd
        
    active_df = pd.DataFrame({'timestamp': timestamp, 'magMean': MAG})
    active_df['magMean'] = active_df['magMean'].replace(0, np.NaN)
    active_df['magMean'] = active_df['magMean'].interpolate(method='values')

    return active_df

def calculate_features(path, activity:str):  ## here
    ### LOAD DATA
    father_data = load_data(path+'Father/')
    mother_data = load_data(path+'Mother/')
    kid_data = load_data(path+'Kid/')
    

    with open(path+ 'Father/' + 'tags.csv', newline='') as f:
        reader = csv.reader(f)
        tags = list(reader)
        tags = list(map(float, list(chain.from_iterable(tags))))

    ## here
    time_intervals={'baseline':[tags[0],tags[1]],
                    '식사':[tags[2],tags[3]],
                    '영상 시청':[tags[4],tags[5]],
                    '보드 게임':[tags[6],tags[7]],
                    '청소':[tags[8],tags[9]]}

    ### Slice Data by activity time
    ts = time_intervals[f'{activity}'][0]
    te = time_intervals[f'{activity}'][1]

    father_eda = father_data[1].loc[(father_data[1].timestamp >= ts) & (father_data[1].timestamp < te),]
    father_eda = father_eda.reset_index(drop=True)
    mother_eda = mother_data[1].loc[(mother_data[1].timestamp >= ts) & (mother_data[1].timestamp < te),]
    mother_eda = mother_eda.reset_index(drop=True)
    kid_eda = kid_data[1].loc[(kid_data[1].timestamp >= ts) & (kid_data[1].timestamp < te),]
    kid_eda = kid_eda.reset_index(drop=True)

    father_ibi = father_data[2].loc[(father_data[2].timestamp >= ts) & (father_data[2].timestamp < te),]
    father_ibi = father_ibi.reset_index(drop=True)
    mother_ibi = mother_data[2].loc[(mother_data[2].timestamp >= ts) & (mother_data[2].timestamp < te),]
    mother_ibi = mother_ibi.reset_index(drop=True)
    kid_ibi = kid_data[2].loc[(kid_data[2].timestamp >= ts) & (kid_data[2].timestamp < te),]
    kid_ibi = kid_ibi.reset_index(drop=True)

    father_acc= father_data[0].loc[(father_data[0].timestamp >= ts) & (father_data[0].timestamp < te),]
    father_acc = father_acc.reset_index(drop=True)
    father_acc['mag'] = np.sqrt(father_acc['x']**2+father_acc['y']**2+father_acc['z']**2)
    mother_acc = mother_data[0].loc[(mother_data[0].timestamp >= ts) & (mother_data[0].timestamp < te),]
    mother_acc = mother_acc.reset_index(drop=True)
    mother_acc['mag'] = np.sqrt(mother_acc['x']**2+mother_acc['y']**2+mother_acc['z']**2)
    kid_acc = kid_data[0].loc[(kid_data[0].timestamp >= ts) & (kid_data[0].timestamp < te),]
    kid_acc = kid_acc.reset_index(drop=True)
    kid_acc['mag'] = np.sqrt(kid_acc['x']**2+kid_acc['y']**2+kid_acc['z']**2)


    ### EDA feature
    father_eda_peaks = count_peaks(father_eda,60)
    mother_eda_peaks = count_peaks(mother_eda,60)
    kid_eda_peaks = count_peaks(kid_eda,60)

    ### HRV feature
    father_HRV_df = calculate_HRV(father_ibi, 60)
    mother_HRV_df = calculate_HRV(mother_ibi, 60)
    kid_HRV_df = calculate_HRV(kid_ibi, 60)

    ### Active feature
    father_active_df = calculate_active(father_acc,60)
    mother_active_df = calculate_active(mother_acc,60)
    kid_active_df = calculate_active(kid_acc,60)

    return [father_eda_peaks, mother_eda_peaks, kid_eda_peaks, father_HRV_df, mother_HRV_df, kid_HRV_df, father_active_df, mother_active_df, kid_active_df]

def calculate_total(path, activities):
    temp = []
    for acv in activities:
        if acv == 'baseline':
            base_dfs = calculate_features(path, acv)
            temp.append(base_dfs)
        temp.append(calculate_features(path, acv))
    
    
    for i in range(len(temp)):
        if i ==0:
            father_total_scr=temp[i][0]
            mother_total_scr=temp[i][1]
            kid_total_scr=temp[i][2]
            father_total_hrv=temp[i][3]
            mother_total_hrv=temp[i][4]
            kid_total_hrv=temp[i][5]
            father_total_act=temp[i][6]
            mother_total_act=temp[i][7]
            kid_total_act=temp[i][8]
        else:
            father_total_scr=pd.concat([father_total_scr,temp[i][0]])
            mother_total_scr=pd.concat([mother_total_scr,temp[i][1]])
            kid_total_scr=pd.concat([kid_total_scr,temp[i][2]])
            father_total_hrv=pd.concat([father_total_hrv,temp[i][3]])
            mother_total_hrv=pd.concat([mother_total_hrv,temp[i][4]])
            kid_total_hrv=pd.concat([kid_total_hrv,temp[i][5]])
            father_total_act=pd.concat([father_total_act,temp[i][6]])
            mother_total_act=pd.concat([mother_total_act,temp[i][7]])
            kid_total_act=pd.concat([kid_total_act,temp[i][8]])
    
    total_dfs = [father_total_scr, mother_total_scr, kid_total_scr, father_total_hrv, mother_total_hrv, kid_total_hrv, father_total_act, mother_total_act, kid_total_act]
    result = []
    for i in range(9):
        a = base_dfs[i].iloc[:,1].mean()
        mad = median_abs_deviation(total_dfs[i].iloc[:,1])
        if i in [3,4,5]:
            lower = total_dfs[i].iloc[:,1].median()-3*mad
            index = total_dfs[i][total_dfs[i].iloc[:,1]<lower].index
            total_dfs[i] = total_dfs[i].drop(index, inplace=False)
            b = total_dfs[i].iloc[:,1].min()
        else:
            upper = total_dfs[i].iloc[:,1].median()+3*mad
            index = total_dfs[i][total_dfs[i].iloc[:,1]>upper].index
            total_dfs[i] = total_dfs[i].drop(index, inplace=False)
            b = total_dfs[i].iloc[:,1].max()
        result.append([a,b])
    return result

def normlize_values(path, activity):
    
    dfs = calculate_features(path, activity)
    vals = calculate_total(path, activities)
    for i in range(9):
        if i in [3,4,5]:
            dfs[i].iloc[:,1] = (dfs[i].iloc[:,1] - vals[i][1])/(vals[i][0]-vals[i][1])
        else:
            dfs[i].iloc[:,1] = (dfs[i].iloc[:,1] - vals[i][0])/(vals[i][1]-vals[i][0])
        dfs[i].iloc[:,1] = dfs[i].iloc[:,1].apply(lambda x: 1 if x >= 1 else x)
        dfs[i].iloc[:,1] = dfs[i].iloc[:,1].apply(lambda x: 0 if x <= 0 else x)
    
    father_eda_peaks = dfs[0]
    mother_eda_peaks = dfs[1]
    kid_eda_peaks = dfs[2]
    father_HRV_df = dfs[3]
    mother_HRV_df = dfs[4]
    kid_HRV_df = dfs[5]
    father_active_df = dfs[6]
    mother_active_df = dfs[7]
    kid_active_df = dfs[8]
    return father_eda_peaks, mother_eda_peaks, kid_eda_peaks, father_HRV_df, mother_HRV_df, kid_HRV_df, father_active_df, mother_active_df, kid_active_df

def arousal_lv(row):

    lower = 0.25
    mid = 0.5
    upper = 0.75
    if (row['peak_per_min'] <= lower):
        return '0: 낮음'
    elif (row['peak_per_min'] > lower)&(row['peak_per_min']<=mid):
        return '1: 적당함'
    elif (row['peak_per_min'] > mid)&(row['peak_per_min']<=upper):
        return '2: 높음'
    else:
        return '3: 아주 높음'

def stress_lv(row):

    lower = 0.25
    mid = 0.5
    upper = 0.75
    
    if (row['RMSSD'] <= lower):
        return '3: 아주 높음'
    elif (row['RMSSD'] > lower)&(row['RMSSD']<=mid):
        return '2: 높음'
    elif (row['RMSSD'] > mid)&(row['RMSSD']<=upper):
        return '1: 적당함'
    else:
        return '0: 낮음'

def active_lv(row):

    lower = 0.25
    mid = 0.5
    upper = 0.75
    if (row['magMean'] <= lower):
        return '0: 낮음'
    elif (row['magMean'] > lower)&(row['magMean']<=mid):
        return '1: 적당함'
    elif (row['magMean'] > mid)&(row['magMean']<=upper):
        return '2: 높음'
    else:
        return '3: 아주 높음'

def convert_to_level(path, activity):
    father_eda_peaks, mother_eda_peaks, kid_eda_peaks, father_HRV_df, mother_HRV_df, kid_HRV_df, father_active_df, mother_active_df, kid_active_df = normlize_values(path, activity)
    ### Emotional aoursal level
    if not father_eda_peaks.empty:
        father_eda_peaks['arousal_lv'] = father_eda_peaks.apply(lambda row: arousal_lv(row), axis=1)
    else:
        father_eda_peaks['arousal_lv'] = None
    father_eda_peaks['datetime'] = pd.to_datetime(father_eda_peaks.timestamp, unit='s')+pd.Timedelta(hours=9)
    father_eda_peaks['datetime'] = father_eda_peaks['datetime'].dt.floor('T')
    
    if not mother_eda_peaks.empty:
        mother_eda_peaks['arousal_lv'] = mother_eda_peaks.apply(lambda row: arousal_lv(row), axis=1)
    else:
        mother_eda_peaks['arousal_lv'] = None
    mother_eda_peaks['datetime'] = pd.to_datetime(mother_eda_peaks.timestamp, unit='s')+pd.Timedelta(hours=9)
    mother_eda_peaks['datetime'] = mother_eda_peaks['datetime'].dt.floor('T')

    if not kid_eda_peaks.empty:
        kid_eda_peaks['arousal_lv'] = kid_eda_peaks.apply(lambda row: arousal_lv(row), axis=1)
    else:
        kid_eda_peaks['arousal_lv'] = None
    kid_eda_peaks['datetime'] = pd.to_datetime(kid_eda_peaks.timestamp, unit='s')+pd.Timedelta(hours=9)
    kid_eda_peaks['datetime'] = kid_eda_peaks['datetime'].dt.floor('T')

    merged_eda_df = pd.merge(father_eda_peaks, mother_eda_peaks, left_on='datetime',right_on='datetime',how='outer')
    merged_eda_df = pd.merge(merged_eda_df, kid_eda_peaks, left_on='datetime',right_on='datetime',how='outer')
    merged_eda_df = merged_eda_df.loc[:,['datetime','arousal_lv_x','arousal_lv_y','arousal_lv']]
    merged_eda_df.rename(columns = {'arousal_lv' : 'arousal_lv_z'}, inplace = True)
    merged_eda_df.fillna('4: 없음', inplace=True)

    ### Stress level
    if not father_HRV_df.empty:
        father_HRV_df['stress_lv'] = father_HRV_df.apply(lambda row: stress_lv(row), axis=1)
    else:
        father_HRV_df['stress_lv'] = None
    father_HRV_df['datetime'] = pd.to_datetime(father_HRV_df.timestamp, unit='s')+pd.Timedelta(hours=9)
    father_HRV_df['datetime'] = father_HRV_df['datetime'].dt.floor('T')  #.dt.strftime('%H:%M')

    if not mother_HRV_df.empty:
        mother_HRV_df['stress_lv'] = mother_HRV_df.apply(lambda row: stress_lv(row), axis=1)
    else:
        mother_HRV_df['stress_lv'] = None
    mother_HRV_df['datetime'] = pd.to_datetime(mother_HRV_df.timestamp, unit='s')+pd.Timedelta(hours=9)
    mother_HRV_df['datetime'] = mother_HRV_df['datetime'].dt.floor('T')

    if not kid_HRV_df.empty:
        kid_HRV_df['stress_lv'] = kid_HRV_df.apply(lambda row: stress_lv(row), axis=1)
    else:
        kid_HRV_df['stress_lv'] = None
    kid_HRV_df['datetime'] = pd.to_datetime(kid_HRV_df.timestamp, unit='s')+pd.Timedelta(hours=9)
    kid_HRV_df['datetime'] = kid_HRV_df['datetime'].dt.floor('T')

    merged_hrv_df = pd.merge(father_HRV_df, mother_HRV_df, left_on='datetime',right_on='datetime',how='outer')
    merged_hrv_df = pd.merge(merged_hrv_df, kid_HRV_df, left_on='datetime',right_on='datetime',how='outer')
    merged_hrv_df = merged_hrv_df.loc[:,['datetime','stress_lv_x','stress_lv_y','stress_lv']]
    merged_hrv_df.rename(columns = {'stress_lv' : 'stress_lv_z'}, inplace = True)
    merged_hrv_df.fillna('4: 없음', inplace=True)

    ### Active Level
    if not father_active_df.empty:
        father_active_df['active_lv'] = father_active_df.apply(lambda row: active_lv(row), axis=1)
    else:
        father_active_df['active_lv'] = None
    father_active_df['datetime'] = pd.to_datetime(father_active_df.timestamp, unit='s')+pd.Timedelta(hours=9)
    father_active_df['datetime'] = father_active_df['datetime'].dt.floor('T')

    if not mother_active_df.empty:
        mother_active_df['active_lv'] = mother_active_df.apply(lambda row: active_lv(row), axis=1)
    else:
        mother_active_df['active_lv'] = None
    mother_active_df['datetime'] = pd.to_datetime(mother_active_df.timestamp, unit='s')+pd.Timedelta(hours=9)
    mother_active_df['datetime'] = mother_active_df['datetime'].dt.floor('T')

    if not kid_active_df.empty:
        kid_active_df['active_lv'] = kid_active_df.apply(lambda row: active_lv(row), axis=1)
    else:
        kid_active_df['active_lv'] = None
    kid_active_df['datetime'] = pd.to_datetime(kid_active_df.timestamp, unit='s')+pd.Timedelta(hours=9)
    kid_active_df['datetime'] = kid_active_df['datetime'].dt.floor('T')

    merged_active_df = pd.merge(father_active_df, mother_active_df, left_on='datetime',right_on='datetime',how='outer')
    merged_active_df = pd.merge(merged_active_df, kid_active_df, left_on='datetime',right_on='datetime',how='outer')
    merged_active_df = merged_active_df.loc[:,['datetime','active_lv_x','active_lv_y','active_lv']]
    merged_active_df.rename(columns = {'active_lv' : 'active_lv_z'}, inplace = True)
    merged_active_df.fillna('4: 없음', inplace=True)

    merged_df = pd.merge(merged_eda_df, merged_hrv_df, left_on='datetime',right_on='datetime',how='outer')
    merged_df = pd.merge(merged_df, merged_active_df, left_on='datetime',right_on='datetime',how='outer')
    merged_df['new'] = np.ones(len(merged_df))
    merged_df.fillna('4: 없음', inplace=True)

    members = ['아빠', '엄마', '아이']
    levels = ['0: 낮음', '1: 적당함', '2: 높음', '3: 아주 높음', '4: 없음']
    ## Aggregate
    father_eda_peaks = pd.DataFrame({'datetime': merged_df.datetime, 'arousal_lv' : merged_df.arousal_lv_x})
    father_eda_peaks['Member'] = '아빠'
    mother_eda_peaks = pd.DataFrame({'datetime': merged_df.datetime, 'arousal_lv' : merged_df.arousal_lv_y})
    mother_eda_peaks['Member'] = '엄마'
    kid_eda_peaks = pd.DataFrame({'datetime': merged_df.datetime, 'arousal_lv' : merged_df.arousal_lv_z})
    kid_eda_peaks['Member'] = '아이'

    concated = pd.concat([father_eda_peaks,mother_eda_peaks,kid_eda_peaks]).loc[:,['Member','arousal_lv']]
    merged_eda_occur = pd.DataFrame({'count':concated.groupby(['Member','arousal_lv']).size()}).reset_index()
    
    for mem in members:
        for lv in levels:
            if not ((merged_eda_occur['Member'] == mem ) & (merged_eda_occur['arousal_lv'] == lv)).any():
                merged_eda_occur = merged_eda_occur.append({'Member': mem, 'arousal_lv': lv, 'count': 0}, ignore_index=True)
    merged_eda_occur = merged_eda_occur.sort_values(['Member', 'arousal_lv'],ascending = [True, True])

    father_HRV_df = pd.DataFrame({'datetime': merged_df.datetime, 'stress_lv' : merged_df.stress_lv_x})
    father_HRV_df['Member'] = '아빠'
    mother_HRV_df = pd.DataFrame({'datetime': merged_df.datetime, 'stress_lv' : merged_df.stress_lv_y})
    mother_HRV_df['Member'] = '엄마'
    kid_HRV_df = pd.DataFrame({'datetime': merged_df.datetime, 'stress_lv' : merged_df.stress_lv_z})
    kid_HRV_df['Member'] = '아이'

    concated = pd.concat([father_HRV_df, mother_HRV_df, kid_HRV_df]).loc[:,['Member','stress_lv']]
    merged_hrv_occur = pd.DataFrame({'count':concated.groupby(['Member','stress_lv']).size()}).reset_index()

    for mem in members:
        for lv in levels:
            if not ((merged_hrv_occur['Member'] == mem ) & (merged_hrv_occur['stress_lv'] == lv)).any():
                merged_hrv_occur = merged_hrv_occur.append({'Member': mem, 'stress_lv': lv, 'count': 0}, ignore_index=True)
    merged_hrv_occur = merged_hrv_occur.sort_values(['Member', 'stress_lv'],ascending = [True, True])

    father_active_df = pd.DataFrame({'datetime': merged_df.datetime, 'active_lv' : merged_df.active_lv_x})
    father_active_df['Member'] = '아빠'
    mother_active_df = pd.DataFrame({'datetime': merged_df.datetime, 'active_lv' : merged_df.active_lv_y})
    mother_active_df['Member'] = '엄마'
    kid_active_df = pd.DataFrame({'datetime': merged_df.datetime, 'active_lv' : merged_df.active_lv_z})
    kid_active_df['Member'] = '아이'

    concated = pd.concat([father_active_df,mother_active_df,kid_active_df]).loc[:,['Member','active_lv']]
    merged_active_occur = pd.DataFrame({'count':concated.groupby(['Member','active_lv']).size()}).reset_index()

    for mem in members:
        for lv in levels:
            if not ((merged_active_occur['Member'] == mem ) & (merged_active_occur['active_lv'] == lv)).any():
                merged_active_occur = merged_active_occur.append({'Member': mem, 'active_lv': lv, 'count': 0}, ignore_index=True)
    merged_active_occur = merged_active_occur.sort_values(['Member', 'active_lv'],ascending = [True, True])
    
    return merged_df, merged_eda_occur, merged_hrv_occur, merged_active_occur
