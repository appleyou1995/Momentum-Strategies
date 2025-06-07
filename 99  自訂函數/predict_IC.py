import pandas as pd
import numpy  as np
import random
import gc

from sklearn.ensemble     import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras.layers         import Dense, Input
from keras.models         import Sequential

import tensorflow  as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 只顯示 ERROR，其他 INFO、WARNING 不顯示


# %%  Function

def predict_IC(df_X, df_Y, 
               month_step, n_loops, seed, 
               window_length, horizon, verbose=True):
    
    def get_year_month(date_str, month_offset):
        year = int(date_str[:4])
        month = int(date_str[5:7])
        month = month + month_offset
        year = year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        return year, month
    
    # 自動取得 start_date_str
    start_date_str = df_X.index[-1]
    
    # 結果列表
    predict_IC = []
    length = len(df_X.columns)
    
    # 設定 Python random 種子
    random.seed(seed)
    
    # 設定 TensorFlow random 種子
    tf.keras.utils.set_random_seed(seed)
    
    for i in range(n_loops):
        
        # 清掉上一次的 TensorFlow graph，防止 memory 累積
        tf.keras.backend.clear_session()
        gc.collect()
    
        year, month = get_year_month(start_date_str, i * month_step)
        
        if verbose:
            print("Current iteration: {}  [ {}-{} ]".format(i, year, str(month).zfill(2)))
        
        ### 樣本切割
        # 全模型通用訓練集
        X = df_X.iloc[:-i-horizon-1,:].values
        Y = df_Y.iloc[horizon:-i-1,:].values
        
        # NN 訓練集
        X_train = df_X.iloc[:-i-window_length-horizon-1,:].values
        Y_train = df_Y.iloc[horizon:-i-window_length-horizon,:].values
        
        # NN 驗證集
        X_validation = df_X.iloc[-i-window_length-horizon-1:-i-horizon-1,:].values
        Y_validation = df_Y.iloc[-i-window_length-horizon:-i-1,:].values
        
        # 測試集 (要預測的當期 IC)
        test_data = df_X.iloc[-i-horizon-1:-i-horizon,:].values
        
        ### 初始化模型
        
        # 線性模型
        reg = LinearRegression()
        
        # 隨機森林
        regr = RandomForestRegressor(max_depth=2, random_state=seed, n_estimators=100)
        
        # NN model generator
        def build_nn_model(layers):
            model = Sequential()
            model.add(Input(shape=(length,)))
            for units in layers:
                model.add(Dense(units=units, activation='relu'))
            model.add(Dense(units=1))
            adam = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=adam, loss='mae')
            return model
        
        model_NN1 = build_nn_model([32])
        model_NN2 = build_nn_model([32,16])
        model_NN3 = build_nn_model([32,16,8])
        model_NN4 = build_nn_model([32,16,8,4])
        model_NN5 = build_nn_model([32,16,8,4,2])
        
        ### 模型訓練
        reg.fit(X,Y)
        regr.fit(X,Y.ravel())
        
        model_NN1.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                      batch_size=32, epochs=10, verbose=0)
        
        model_NN2.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                      batch_size=32, epochs=10, verbose=0)
        
        model_NN3.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                      batch_size=32, epochs=10, verbose=0)
        
        model_NN4.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                      batch_size=32, epochs=10, verbose=0)
        
        model_NN5.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                      batch_size=32, epochs=10, verbose=0)
        
        ### 預測結果
        pred     = reg.predict(test_data)[0][0]
        predregr = regr.predict(test_data)[0]
        prednn1  = model_NN1.predict(test_data)[0][0].astype(np.float64)
        prednn2  = model_NN2.predict(test_data)[0][0].astype(np.float64)
        prednn3  = model_NN3.predict(test_data)[0][0].astype(np.float64)
        prednn4  = model_NN4.predict(test_data)[0][0].astype(np.float64)
        prednn5  = model_NN5.predict(test_data)[0][0].astype(np.float64)
        
        ### 加入結果
        predict_IC.append((year, month, pred, predregr, prednn1, prednn2, prednn3, prednn4, prednn5))
    
    # 輸出 DataFrame
    df_predict_IC = pd.DataFrame(predict_IC, 
                                 columns=['year', 'month', 'Linear', 'RandomForest', 
                                          'NN1', 'NN2', 'NN3', 'NN4', 'NN5'])
    
    return df_predict_IC
