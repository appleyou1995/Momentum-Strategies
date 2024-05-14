# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:58:45 2023

@author: asus
"""
#1個月動能資料整理，機器學習往回預測100次IC值
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
# 將日期倒序排列
log_returns_reverse = log_returns.T
log_returns_reverse = log_returns_reverse.sort_index(ascending=False)
# 將 DataFrame 寫入 CSV 檔案
log_returns_reverse.to_csv('D:/project2/logreturn/log_returns.csv', index=True, encoding='utf-8')

#1個月動能
#複製log_return
log_returns1 = log_returns.copy()

# 計算每個月每間公司的一個月動能(從1995-02開始有值)(ln(t/t-1))
mom1 = np.log( df_new / df_new.shift(axis=1))

#比對log_returns1和mom1相同時間是否都有值，若其中一個沒有，則另一個也改為NaN
for i in range(len(log_returns1.index)):
    for j in range(len(log_returns1.columns)):
        if np.isnan(log_returns1.iloc[i, j]) or np.isnan(mom1.iloc[i, j]):
            log_returns1.iloc[i, j] = np.nan
            mom1.iloc[i, j] = np.nan

# 刪除整行資料都是 NaN 的行
log_returns1_new = log_returns1.copy()
log_returns1_new.dropna(axis=1, how='all', inplace=True)
mom1_new = mom1.copy()
mom1_new.dropna(axis=1, how='all', inplace=True)

# 將所有NaN的值改為空字串           
log_returns1_new = log_returns1_new.fillna('')
mom1_new = mom1_new.fillna('')

# 計算 log_returns1_new 和 mom1_new 中每行的平均值
log_returns1_new_mean = log_returns1_new.apply(lambda row: row[row!=''].astype(float).mean(), axis=0)
mom1_new_mean = mom1_new.apply(lambda row: row[row!=''].astype(float).mean(), axis=0)

# 將 log_returns1_new 和 mom1_new 中的空值填入各自的平均值
for i in range(len(log_returns1_new.index)):
    for j in range(len(log_returns1_new.columns)):
        if log_returns1_new.iloc[i, j] == '':
            log_returns1_new.iloc[i, j] = log_returns1_new_mean[j]
            mom1_new.iloc[i, j] = mom1_new_mean[j]
            
## 將 log_returns_new 和 mom1_new 轉換為浮點數格式
log_returns1_new = log_returns1_new.apply(pd.to_numeric, errors='coerce')
mom1_new = mom1_new.apply(pd.to_numeric, errors='coerce')

# 對 log_returns1_new 和 mom1_new 分別進行Rank排序
log_returns1_rank = log_returns1_new.copy()
log_returns1_rank = log_returns1_rank.rank(axis=1, na_option='keep')
mom1_rank = mom1_new.copy()
mom1_rank = mom1_rank.rank(axis=1, na_option='keep') #Why出現小數點，有相同股價

# 將日期倒序排列
mom1_rank_reverse = mom1_rank.T
mom1_rank_reverse = mom1_rank_reverse.sort_index(ascending=False)

# 將 DataFrame 寫入 CSV 檔案
mom1_rank_reverse.to_csv('D:/project2/Rank/mom1_rank.csv', index=True, encoding='utf-8')

# 計算IC1值
corr1_list = []  # 用於存儲所有相關係數的列表
for month in log_returns1_rank.columns:
    log_returns1_monthly = log_returns1_rank[month]  # 選擇當月的log_returns_rank向量
    mom1_monthly = mom1_rank[month]  # 選擇當月的mom1_rank向量
    corr1 = np.corrcoef(log_returns1_monthly, mom1_monthly)[0, 1]  # 計算相關係數
    corr1_list.append(corr1)

IC1 = pd.DataFrame({'IC1': corr1_list}, index=log_returns1_rank.columns)  # 將相關係數轉換為DataFrame
IC1.to_csv('D:/project2/IC_original/IC1.csv', index=True, encoding='utf-8')
# 將動能資料轉置(Index為時間，column為股票代號，IC已為此格式不須轉置)
mom1_new = mom1_new.T

# 模型X、Y
df_x_mom1 = mom1_new
df_y_mom1 = IC1

# 設定迴圈次數
n_loops = 100

# 設定初始月份
start_month = 10
start_year = 2022

# 設定每次迴圈的間隔月份
month_step = -1

# 設定儲存結果的列表
results_IC1 = []

# 執行迴圈 #Why只能從2022/10往回(2022/11讀不到資料)
for i in range(n_loops):
    month = start_month + i * month_step
    year = start_year + month // 12
    month %= 12
    # 設定訓練集
    #mom1值(1995/2-2022/9，共320個月) 
    X_mom1 = df_x_mom1.iloc[:-i-2,:].values
    #mom1 IC值(1995/2-2022/10，共320個月)
    Y_mom1 = df_y_mom1.iloc[1:-i-1,:].values
    
    #設定NN1~NN5訓練集
    #mom1值(1995/2-2014/9，共224個月)
    X_mom1_train = df_x_mom1.iloc[:-i-98,:].values
    #mom1 IC值(1995/3-2014/10，共224個月)
    Y_mom1_train = df_y_mom1.iloc[1:-i-97,:].values

    #設定NN1~NN5驗證集
    ##mom1值(2014/10 -2022/9，共96個月)
    X_mom1_validation = df_x_mom1.iloc[-i-98:-2,:].values
    #mom1 IC值(2014/11-2022/10，共96個月)
    Y_mom1_validation = df_y_mom1.iloc[-i-97:-1,:].values
    
    # 設定測試集
    #2022/10的mom1
    test_data_mom1 = df_x_mom1.iloc[-i-2:-i-1,:].values

    # 建立模型
    reg = LinearRegression() #線性模型
    regr = RandomForestRegressor(max_depth=2, random_state=42, n_estimators=100) #隨機森林建模
    #NN1建模
    model_mom1_NN1 = Sequential()
    model_mom1_NN1.add(Dense(units=32,activation='relu',input_dim=4983))
    model_mom1_NN1.add(Dense(units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom1_NN1.compile(optimizer=adam,loss='mae')
    #NN2建模
    model_mom1_NN2 = Sequential()
    model_mom1_NN2.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom1_NN2.add(Dense (units=16, activation='relu'))
    model_mom1_NN2.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom1_NN2.compile(optimizer=adam,loss='mae')
    #NN3建模
    model_mom1_NN3 = Sequential()
    model_mom1_NN3.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom1_NN3.add(Dense (units=16, activation='relu'))
    model_mom1_NN3.add(Dense (units=8, activation='relu'))
    model_mom1_NN3.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom1_NN3.compile(optimizer=adam,loss='mae')
    #NN4建模
    model_mom1_NN4 = Sequential()
    model_mom1_NN4.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom1_NN4.add(Dense (units=16, activation='relu'))
    model_mom1_NN4.add(Dense (units=8, activation='relu'))
    model_mom1_NN4.add(Dense (units=4, activation='relu'))
    model_mom1_NN4.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom1_NN4.compile(optimizer=adam,loss='mae')
    #NN5建模
    model_mom1_NN5 = Sequential()
    model_mom1_NN5.add(Dense (units=32, activation='relu', input_dim=4983))
    model_mom1_NN5.add(Dense (units=16, activation='relu'))
    model_mom1_NN5.add(Dense (units=8, activation='relu'))
    model_mom1_NN5.add(Dense (units=4, activation='relu'))
    model_mom1_NN5.add(Dense (units=2, activation='relu'))
    model_mom1_NN5.add(Dense (units=1))
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_seed = 42
    tf.random.set_seed(my_seed) #讓種子結果相同
    model_mom1_NN5.compile(optimizer=adam,loss='mae')

    # 丟入數據訓練
    reg.fit(X_mom1,Y_mom1)
    regr.fit(X_mom1,Y_mom1.ravel())
    model_mom1_NN1.fit(X_mom1_train,Y_mom1_train,validation_data=(X_mom1_validation,Y_mom1_validation),batch_size=32,epochs=10)
    model_mom1_NN2.fit(X_mom1_train,Y_mom1_train,validation_data=(X_mom1_validation,Y_mom1_validation),batch_size=32,epochs=10)
    model_mom1_NN3.fit(X_mom1_train,Y_mom1_train,validation_data=(X_mom1_validation,Y_mom1_validation),batch_size=32,epochs=10)
    model_mom1_NN4.fit(X_mom1_train,Y_mom1_train,validation_data=(X_mom1_validation,Y_mom1_validation),batch_size=32,epochs=10)
    model_mom1_NN5.fit(X_mom1_train,Y_mom1_train,validation_data=(X_mom1_validation,Y_mom1_validation),batch_size=32,epochs=10)

    # 預測結果
    pred = reg.predict(test_data_mom1)[0][0]
    predregr = regr.predict(test_data_mom1)[0]
    prednn1 = model_mom1_NN1.predict(test_data_mom1)[0]
    prednn2 = model_mom1_NN2.predict(test_data_mom1)[0]
    prednn3 = model_mom1_NN3.predict(test_data_mom1)[0]
    prednn4 = model_mom1_NN4.predict(test_data_mom1)[0]
    prednn5 = model_mom1_NN5.predict(test_data_mom1)[0]
    # 將結果加入到列表中
    results_IC1.append((year, month, pred, predregr, prednn1, prednn2, prednn3 ,prednn4, prednn5))

    #將月份為0的資料修正為前一年的12月份
    for i in range(len(results_IC1)):
        if results_IC1[i][1] == 0:
            results_IC1[i] = (results_IC1[i][0]-1, 12, results_IC1[i][2], results_IC1[i][3], results_IC1[i][4], results_IC1[i][5], results_IC1[i][6], results_IC1[i][7], results_IC1[i][8])
         
    # 將 results 轉換為 DataFrame
    df_results_IC1 = pd.DataFrame(results_IC1, columns=['year', 'month', 'Linear_mom1', 'RandomForest_mom1', 'NN1_mom1', 'NN2_mom1', 'NN3_mom1', 'NN4_mom1', 'NN5_mom1'])
    
    # 將包含數字的字串轉換為浮點數
    df_results_IC1['NN1_mom1'] = df_results_IC1['NN1_mom1'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC1['NN2_mom1'] = df_results_IC1['NN2_mom1'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC1['NN3_mom1'] = df_results_IC1['NN3_mom1'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC1['NN4_mom1'] = df_results_IC1['NN4_mom1'].apply(lambda x: float(x) if len(x)>0 else None)
    df_results_IC1['NN5_mom1'] = df_results_IC1['NN5_mom1'].apply(lambda x: float(x) if len(x)>0 else None)

    # 將 DataFrame 寫入 CSV 檔案
    df_results_IC1.to_csv('D:/project2/IC/results_IC1.csv', index=False, encoding='utf-8')
