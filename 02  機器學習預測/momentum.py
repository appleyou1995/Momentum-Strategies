import os
import sys
import pandas as pd
import numpy  as np

from sklearn              import ensemble
from sklearn.ensemble     import RandomForestRegressor, GradientBoostingRegressor
from xgboost              import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm          import SVR
from keras.layers         import Dense, Input, Activation
from keras.models         import Sequential
from keras.models         import load_model
from tensorflow.keras     import optimizers

import tensorflow  as tf

# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)


# %%  Mac 資料夾路徑

Path_Mac = '/Users/irisyu/Library/CloudStorage/GoogleDrive-jouping.yu@gmail.com/'
Path_dir = os.path.join(Path_Mac, Path_PaperFolder)


# %%  Input and Output Path

Path_Input  = os.path.join(Path_dir, 'Data/')
Path_Output = os.path.join(Path_dir, 'Data/')


# %%  Import data

df_price      = pd.read_csv(os.path.join(Path_Input, 'PRC.csv')        , index_col='PERMNO')
df_log_return = pd.read_csv(os.path.join(Path_Input, 'log_returns.csv'), index_col='PERMNO')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from Information_Coefficient import Information_Coefficient


# %%  計算 momentum、IC 值

mom_01m, IC_01m = Information_Coefficient(df_price, df_log_return, 1)
mom_06m, IC_06m = Information_Coefficient(df_price, df_log_return, 6)
mom_12m, IC_12m = Information_Coefficient(df_price, df_log_return, 12)
mom_36m, IC_36m = Information_Coefficient(df_price, df_log_return, 36)
mom_60m, IC_60m = Information_Coefficient(df_price, df_log_return, 60)


# %%  匯出表格

mom_01m.to_csv(Path_Output+'mom_01m.csv', index=True, encoding='utf-8')
mom_06m.to_csv(Path_Output+'mom_06m.csv', index=True, encoding='utf-8')
mom_12m.to_csv(Path_Output+'mom_12m.csv', index=True, encoding='utf-8')
mom_36m.to_csv(Path_Output+'mom_36m.csv', index=True, encoding='utf-8')
mom_60m.to_csv(Path_Output+'mom_60m.csv', index=True, encoding='utf-8')

IC_01m.to_csv(Path_Output+'IC_01m.csv', index=True, encoding='utf-8')
IC_06m.to_csv(Path_Output+'IC_06m.csv', index=True, encoding='utf-8')
IC_12m.to_csv(Path_Output+'IC_12m.csv', index=True, encoding='utf-8')
IC_36m.to_csv(Path_Output+'IC_36m.csv', index=True, encoding='utf-8')
IC_60m.to_csv(Path_Output+'IC_60m.csv', index=True, encoding='utf-8')


# %%  模型設定

# 模型 X、Y
df_x_mom1 = mom_01m
df_y_mom1 = IC_01m

# 設定迴圈次數
n_loops = 100

# 設定初始月份
start_month = 10
start_year  = 2022

# 設定每次迴圈的間隔月份
month_step = -1

# 設定隨機種子以確保結果的可重現性
my_seed = 42

# 設定儲存結果的列表
predict_IC1 = []
    
    
# %%  建立模型 

# 線性模型
reg = LinearRegression()

# 隨機森林
regr = RandomForestRegressor(max_depth=2, random_state=42, n_estimators=100)


# 神經網路
tf.random.set_seed(my_seed)                                                    # 設定隨機種子以確保結果的可重現性

# NN1：定義和編譯模型
model_mom1_NN1 = Sequential([
    Input(shape=(4905,)),                                                      # 增加輸入層來指定輸入形狀
    Dense(units=32, activation='relu'),                                        # 增加第一個隱藏層
    Dense(units=1)                                                             # 增加輸出層
])
adam = tf.keras.optimizers.Adam(learning_rate=0.001)                           # 設置 Adam 優化器
model_mom1_NN1.compile(optimizer=adam, loss='mae')                             # 編譯模型

# NN2：定義和編譯模型
model_mom1_NN2 = Sequential([
    Input(shape=(4905,)),                                                      # 增加輸入層來指定輸入形狀
    Dense(units=32, activation='relu'),                                        # 增加第一個隱藏層
    Dense(units=16, activation='relu'),                                        # 增加第二個隱藏層
    Dense(units=1)                                                             # 增加輸出層
])
adam = tf.keras.optimizers.Adam(learning_rate=0.001)                           # 設置 Adam 優化器
model_mom1_NN2.compile(optimizer=adam, loss='mae')                             # 編譯模型

# NN3 建模
model_mom1_NN3 = Sequential([
    Input(shape=(4905,)),                                                      # 增加輸入層來指定輸入形狀
    Dense(units=32, activation='relu'),                                        # 增加第一個隱藏層
    Dense(units=16, activation='relu'),                                        # 增加第二個隱藏層
    Dense(units=8, activation='relu'),                                         # 增加第三個隱藏層
    Dense(units=1)                                                             # 增加輸出層
])
adam = tf.keras.optimizers.Adam(learning_rate=0.001)                           # 設置 Adam 優化器
model_mom1_NN3.compile(optimizer=adam, loss='mae')                             # 編譯模型

# NN4 建模
model_mom1_NN4 = Sequential([
    Input(shape=(4905,)),                                                      # 增加輸入層來指定輸入形狀
    Dense(units=32, activation='relu'),                                        # 增加第一個隱藏層
    Dense(units=16, activation='relu'),                                        # 增加第二個隱藏層
    Dense(units=8, activation='relu'),                                         # 增加第三個隱藏層
    Dense(units=4, activation='relu'),                                         # 增加第四個隱藏層
    Dense(units=1)                                                             # 增加輸出層
])
adam = tf.keras.optimizers.Adam(learning_rate=0.001)                           # 設置 Adam 優化器
model_mom1_NN4.compile(optimizer=adam, loss='mae')                             # 編譯模型

# NN5 建模
model_mom1_NN5 = Sequential([
    Input(shape=(4905,)),                                                      # 增加輸入層來指定輸入形狀
    Dense(units=32, activation='relu'),                                        # 增加第一個隱藏層
    Dense(units=16, activation='relu'),                                        # 增加第二個隱藏層
    Dense(units=8, activation='relu'),                                         # 增加第三個隱藏層
    Dense(units=4, activation='relu'),                                         # 增加第四個隱藏層
    Dense(units=2, activation='relu'),                                         # 增加第五個隱藏層
    Dense(units=1)                                                             # 增加輸出層
])
adam = tf.keras.optimizers.Adam(learning_rate=0.001)                           # 設置 Adam 優化器
model_mom1_NN5.compile(optimizer=adam, loss='mae')                             # 編譯模型


# %%  迴圈

for i in range(n_loops):

    month = start_month + i * month_step
    year = start_year + month // 12
    
    # 調整月份
    if month % 12 == 0:
        month = 12
        year -= 1
    else:
        month %= 12
    
    print(str(year) + "-" + str(month))

    ### 樣本切割
    
    # 設定訓練集 (待確認)
    # mom1 值
    X_mom1 = df_x_mom1.iloc[:-i-2,:].values                                    # (1995-02 至 2022-09，共 332 個月) 
    # mom1 IC 值
    Y_mom1 = df_y_mom1.iloc[1:-i-1,:].values                                   # (1995-03 至 2022-10，共 332 個月)
    
    # 設定 NN1 ~ NN5 訓練集 (待確認)
    # mom1 值
    X_mom1_train = df_x_mom1.iloc[:-i-98,:].values                             # (1995-02 至 2014-09，共 236 個月)
    # mom1 IC 值
    Y_mom1_train = df_y_mom1.iloc[1:-i-97,:].values                            # (1995-03 至 2014-10，共 236 個月)

    # 設定 NN1 ~ NN5 驗證集
    # mom1 值   
    X_mom1_validation = df_x_mom1.iloc[-i-98:-2,:].values                      # (2014-10 至 2022-09，共 96 個月)
    # mom1 IC 值
    Y_mom1_validation = df_y_mom1.iloc[-i-97:-1,:].values                      # (2014-11 至 2022-10，共 96 個月)
    
    # 設定測試集
    # 2022-10 的 mom1
    test_data_mom1 = df_x_mom1.iloc[-i-2:-i-1,:].values


    ### 丟入數據訓練
    reg.fit(X_mom1,Y_mom1)
    regr.fit(X_mom1,Y_mom1.ravel())
    
    model_mom1_NN1.fit(
        X_mom1_train, Y_mom1_train,                                            # 訓練數據和標籤
        validation_data=(X_mom1_validation, Y_mom1_validation),                # 驗證數據和標籤
        batch_size=32,                                                         # 模型在每次更新權重時會使用 32 個樣本
        epochs=10                                                              # 訓練 10 個迭代
    )
    
    model_mom1_NN2.fit(
        X_mom1_train, Y_mom1_train,
        validation_data=(X_mom1_validation, Y_mom1_validation),
        batch_size=32,
        epochs=10
    )
    
    model_mom1_NN3.fit(
        X_mom1_train, Y_mom1_train,
        validation_data=(X_mom1_validation, Y_mom1_validation),
        batch_size=32,
        epochs=10
    )
    
    model_mom1_NN4.fit(
        X_mom1_train, Y_mom1_train,
        validation_data=(X_mom1_validation, Y_mom1_validation),
        batch_size=32,
        epochs=10
    )
    
    model_mom1_NN5.fit(
        X_mom1_train, Y_mom1_train,
        validation_data=(X_mom1_validation, Y_mom1_validation),
        batch_size=32,
        epochs=10
    )

    
    # 預測結果
    pred = reg.predict(test_data_mom1)[0][0]
    predregr = regr.predict(test_data_mom1)[0]
    prednn1 = model_mom1_NN1.predict(test_data_mom1)[0][0].astype(np.float64)
    prednn2 = model_mom1_NN2.predict(test_data_mom1)[0][0].astype(np.float64)
    prednn3 = model_mom1_NN3.predict(test_data_mom1)[0][0].astype(np.float64)
    prednn4 = model_mom1_NN4.predict(test_data_mom1)[0][0].astype(np.float64)
    prednn5 = model_mom1_NN5.predict(test_data_mom1)[0][0].astype(np.float64)
    
    # 將結果加入到列表中
    predict_IC1.append((year, month, pred, predregr, prednn1, prednn2, prednn3 ,prednn4, prednn5))



# 將 results 轉換為 DataFrame
df_predict_IC1 = pd.DataFrame(predict_IC1, 
                              columns=['year', 'month', 'Linear_mom1', 'RandomForest_mom1', 
                                       'NN1_mom1', 'NN2_mom1', 'NN3_mom1', 'NN4_mom1', 'NN5_mom1'])
    
df_predict_IC1.to_csv(Path_Output+'predict_IC1.csv', index=False, encoding='utf-8')
