# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:37:50 2023

@author: asus
"""
#12個月動能資料整理，機器學習往回預測100次IC值
import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as pit
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras import optimizers
import tensorflow as tf

# 檔案夾路徑
file_dir = "D:/project2/data"

# 初始化資料框
df_merged = pd.DataFrame()

# 讀取每個檔案的 1、2 和 47 行資料並合併
for filename in os.listdir(file_dir):
    if filename.endswith(".csv"):
        # 讀取指定行的資料
        df = pd.read_csv(os.path.join(file_dir, filename), usecols=[0, 1, 5, 47])
        # 將第三行的數值轉換為浮點數格式
        df.iloc[:, 3] = df.iloc[:, 3].astype(float)
        # 取第三行數值的絕對值
        df.iloc[:, 3] = df.iloc[:, 3].abs()
        # 將讀取到的資料框與已經讀取的資料框合併
        df_merged = pd.concat([df_merged, df], axis=0)
#空白補值
df_merged['SICCD'].fillna(method='ffill', inplace=True)
# 刪除 'SICCD' 欄位內非數字值的整列
df_merged = df_merged[pd.to_numeric(df_merged['SICCD'], errors='coerce').notnull()]
df_merged.iloc[:, 2] = df_merged.iloc[:, 2].astype(int)
# 刪除SICCD值在4900到4999和6000到6999之間的整列
df_merged = df_merged[(df_merged['SICCD'] <= 4900) | (df_merged['SICCD'] >= 4999)]
df_merged = df_merged[(df_merged['SICCD'] <= 6000) | (df_merged['SICCD'] >= 6999)]

# 將df_merged轉換為以PERMNO為索引，日期為列名的新資料框
df_new = pd.pivot_table(df_merged, index='PERMNO', columns='date', values='PRC')

# 將df_new的column改成DatetimeIndex
df_new.columns = pd.to_datetime(df_new.columns).strftime("%Y-%m")

# 其中2022-10、2022-11、2022-12錯位
# 將指定欄位名稱移動到最後面
col_names = list(df_new.columns)
col_names.remove('2022-10')
col_names.remove('2022-11')
col_names.remove('2022-12')
col_names += ['2022-10', '2022-11', '2022-12']
df_new = df_new.reindex(columns=col_names)

# 2022/12已下市的股票刪除
df_new = df_new.dropna(subset=df_new.columns[-2:-1], how='any')

# 將所有股價轉為浮點數
df_new = df_new.apply(pd.to_numeric, errors='coerce')

# 計算每間公司每個月的logreturn(從1995-01開始有值)(ln(t+1/t))
log_returns = np.log( df_new.shift(-1, axis=1)/df_new)

#12個月動能
#複製log_return
log_returns12 = log_returns.copy()

# 計算每個月每間公司的12個月動能(從1996-01開始有值)(ln(t/t-12))
mom12 = np.log( df_new / df_new.shift(12 , axis=1))

#比對log_returns12和mom12相同時間是否都有值，若其中一個沒有，則另一個也改為NaN
for i in range(len(log_returns12.index)):
    for j in range(len(log_returns12.columns)):
        if np.isnan(log_returns12.iloc[i, j]) or np.isnan(mom12.iloc[i, j]):
            log_returns12.iloc[i, j] = np.nan
            mom12.iloc[i, j] = np.nan

# 刪除整行資料都是 NaN 的行
log_returns12_new = log_returns12.copy()
log_returns12_new.dropna(axis=1, how='all', inplace=True)
mom12_new = mom12.copy()
mom12_new.dropna(axis=1, how='all', inplace=True)

# 將所有NaN的值改為空字串           
log_returns12_new = log_returns12_new.fillna('')
mom12_new = mom12_new.fillna('')

# 計算 log_returns12_new 和 mom12_new 中每行的平均值
log_returns12_new_mean = log_returns12_new.apply(lambda row: row[row!=''].astype(float).mean(), axis=0)
mom12_new_mean = mom12_new.apply(lambda row: row[row!=''].astype(float).mean(), axis=0)

# 將 log_returns12_new 和 mom12_new 中的空值填入各自的平均值
for i in range(len(log_returns12_new.index)):
    for j in range(len(log_returns12_new.columns)):
        if log_returns12_new.iloc[i, j] == '':
            log_returns12_new.iloc[i, j] = log_returns12_new_mean[j]
            mom12_new.iloc[i, j] = mom12_new_mean[j]
            
## 將 log_returns_new 和 mom1_new 轉換為浮點數格式
log_returns12_new = log_returns12_new.apply(pd.to_numeric, errors='coerce')
mom12_new = mom12_new.apply(pd.to_numeric, errors='coerce')

# 對 log_returns12_new 和 mom12_new 分別進行Rank排序
log_returns12_rank = log_returns12_new.copy()
log_returns12_rank = log_returns12_rank.rank(axis=1, na_option='keep')
mom12_rank = mom12_new.copy()
mom12_rank = mom12_rank.rank(axis=1, na_option='keep') #Why出現小數點，有相同股價

# 將日期倒序排列
mom12_rank_reverse = mom12_rank.T
mom12_rank_reverse = mom12_rank_reverse.sort_index(ascending=False)

# 將 DataFrame 寫入 CSV 檔案
mom12_rank_reverse.to_csv('D:/project2/Rank/mom12_rank.csv', index=True, encoding='utf-8')
                          
# 計算IC1值
corr12_list = []  # 用於存儲所有相關係數的列表
for month in log_returns12_rank.columns:
    log_returns12_monthly = log_returns12_rank[month]  # 選擇當月的log_returns12_rank向量
    mom12_monthly = mom12_rank[month]  # 選擇當月的mom12_rank向量
    corr12 = np.corrcoef(log_returns12_monthly, mom12_monthly)[0, 1]  # 計算相關係數
    corr12_list.append(corr12)

IC12 = pd.DataFrame({'IC12': corr12_list}, index=log_returns12_rank.columns)  # 將相關係數轉換為DataFrame
IC12.to_csv('D:/project2/IC_original/IC12.csv', index=True, encoding='utf-8')
# 將動能資料轉置(Index為時間，column為股票代號，IC已為此格式不須轉置)
mom12_new = mom12_new.T

# 模型X、Y
df_x_mom12 = mom12_new
df_y_mom12 = IC12

# 設定迴圈次數
n_loops = 100

# 設定初始月份
start_month = 10
start_year = 2022

# 設定每次迴圈的間隔月份
month_step = -1

# 設定儲存結果的列表
results_IC12 = []

# 執行迴圈 #Why只能從2022/10往回(2022/11讀不到資料)
for i in range(n_loops):
    month = start_month + i * month_step
    year = start_year + month // 12
    month %= 12
    # 設定訓練集
    #mom1值(1995/2-2022/9，共320個月) 
    X_mom12 = df_x_mom12.iloc[:-i-2,:].values
    #mom1 IC值(1995/2-2022/10，共320個月)
    Y_mom12 = df_y_mom12.iloc[1:-i-1,:].values
    
    #設定NN1~NN5訓練集
    #mom1值(1995/2-2014/9，共224個月)
    X_mom12_train = df_x_mom12.iloc[:-i-98,:].values
    #mom1 IC值(1995/3-2014/10，共224個月)
    Y_mom12_train = df_y_mom12.iloc[1:-i-97,:].values

    #設定NN1~NN5驗證集
    ##mom1值(2014/10 -2022/9，共96個月)
    X_mom12_validation = df_x_mom12.iloc[-i-98:-2,:].values
    #mom1 IC值(2014/11-2022/10，共96個月)
    Y_mom12_validation = df_y_mom12.iloc[-i-97:-1,:].values
    
    # 設定測試集
    #2022/10的mom1
    test_data_mom12 = df_x_mom12.iloc[-i-2:-i-1,:].values

    #建立模型
    reg = LinearRegression() #回歸建模
    regr = RandomForestRegressor(max_depth=2, random_state=42, n_estimators=100) #隨機森林建模
    #NN1建模
    model_mom12_NN1 = Sequential()
    model_mom12_NN1.add(Dense(units=32,activation='relu',input_dim=4983))
    model_mom12_NN1.add(Dense(units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom12_NN1.compile(optimizer=adam,loss='mae')
    #NN2建模
    model_mom12_NN2 = Sequential()
    model_mom12_NN2.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom12_NN2.add(Dense (units=16, activation='relu'))
    model_mom12_NN2.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom12_NN2.compile(optimizer=adam,loss='mae')
    #NN3建模
    model_mom12_NN3 = Sequential()
    model_mom12_NN3.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom12_NN3.add(Dense (units=16, activation='relu'))
    model_mom12_NN3.add(Dense (units=8, activation='relu'))
    model_mom12_NN3.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom12_NN3.compile(optimizer=adam,loss='mae')
    #NN4建模
    model_mom12_NN4 = Sequential()
    model_mom12_NN4.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom12_NN4.add(Dense (units=16, activation='relu'))
    model_mom12_NN4.add(Dense (units=8, activation='relu'))
    model_mom12_NN4.add(Dense (units=4, activation='relu'))
    model_mom12_NN4.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom12_NN4.compile(optimizer=adam,loss='mae')
    #NN5建模
    model_mom12_NN5 = Sequential()
    model_mom12_NN5.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom12_NN5.add(Dense (units=16, activation='relu'))
    model_mom12_NN5.add(Dense (units=8, activation='relu'))
    model_mom12_NN5.add(Dense (units=4, activation='relu'))
    model_mom12_NN5.add(Dense (units=2, activation='relu'))
    model_mom12_NN5.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom12_NN5.compile(optimizer=adam,loss='mae')
    
    #丟入數據訓練
    reg.fit(X_mom12,Y_mom12)
    regr.fit(X_mom12,Y_mom12)
    model_mom12_NN1.fit(X_mom12_train,Y_mom12_train,validation_data=(X_mom12_validation,Y_mom12_validation),batch_size=32,epochs=10)
    model_mom12_NN2.fit(X_mom12_train,Y_mom12_train,validation_data=(X_mom12_validation,Y_mom12_validation),batch_size=32,epochs=10)
    model_mom12_NN3.fit(X_mom12_train,Y_mom12_train,validation_data=(X_mom12_validation,Y_mom12_validation),batch_size=32,epochs=10)
    model_mom12_NN4.fit(X_mom12_train,Y_mom12_train,validation_data=(X_mom12_validation,Y_mom12_validation),batch_size=32,epochs=10)
    model_mom12_NN5.fit(X_mom12_train,Y_mom12_train,validation_data=(X_mom12_validation,Y_mom12_validation),batch_size=32,epochs=10)
    
    # 預測結果
    pred = reg.predict(test_data_mom12)[0][0]
    predregr = regr.predict(test_data_mom12)[0]
    prednn1 = model_mom12_NN1.predict(test_data_mom12)[0]
    prednn2 = model_mom12_NN2.predict(test_data_mom12)[0]
    prednn3 = model_mom12_NN3.predict(test_data_mom12)[0]
    prednn4 = model_mom12_NN4.predict(test_data_mom12)[0]
    prednn5 = model_mom12_NN5.predict(test_data_mom12)[0]
    # 將結果加入到列表中
    results_IC12.append((year, month, pred, predregr, prednn1, prednn2, prednn3 ,prednn4, prednn5))
    
    #將月份為0的資料修正為前一年的12月份
    for i in range(len(results_IC12)):
        if results_IC12[i][1] == 0:
            results_IC12[i] = (results_IC12[i][0]-1, 12, results_IC12[i][2], results_IC12[i][3], results_IC12[i][4], results_IC12[i][5], results_IC12[i][6], results_IC12[i][7], results_IC12[i][8])
            

    # 將 results 轉換為 DataFrame
    df_results_IC12 = pd.DataFrame(results_IC12, columns=['year', 'month', 'Linear_mom12', 'RandomForest_mom12', 'NN1_mom12', 'NN2_mom12', 'NN3_mom12', 'NN4_mom12', 'NN5_mom12'])
    
    # 將包含數字的字串轉換為浮點數
    df_results_IC12['NN1_mom12'] = df_results_IC12['NN1_mom12'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC12['NN2_mom12'] = df_results_IC12['NN2_mom12'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC12['NN3_mom12'] = df_results_IC12['NN3_mom12'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC12['NN4_mom12'] = df_results_IC12['NN4_mom12'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC12['NN5_mom12'] = df_results_IC12['NN5_mom12'].apply(lambda x: float(x) if len(x)>0 else None)

    # 將 DataFrame 寫入 CSV 檔案
    df_results_IC12.to_csv('D:/project2/IC/results_IC12.csv', index=False, encoding='utf-8')
