import pandas as pd
import numpy  as np

from sklearn.ensemble     import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras.layers         import Dense, Input
from keras.models         import Sequential

import tensorflow  as tf


# %%


def predict_IC(df_X, df_Y, start_year, start_month, month_step, n_loops, seed):
    
    predict_IC = []
    length = len(df_X.columns)
    
    ### 建立模型 
    
    # 線性模型
    reg = LinearRegression()
    
    # 隨機森林
    regr = RandomForestRegressor(max_depth=2, random_state=42, n_estimators=100)
    
    
    # 神經網路
    tf.random.set_seed(seed)                                                   # 設定隨機種子以確保結果的可重現性
    
    # NN1：定義和編譯模型
    model_NN1 = Sequential([
        Input(shape=(length,)),                                                # 增加輸入層來指定輸入形狀
        Dense(units=32, activation='relu'),                                    # 增加第一個隱藏層
        Dense(units=1)                                                         # 增加輸出層
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)                       # 設置 Adam 優化器
    model_NN1.compile(optimizer=adam, loss='mae')                              # 編譯模型
    
    # NN2：定義和編譯模型
    model_NN2 = Sequential([
        Input(shape=(length,)),                                                # 增加輸入層來指定輸入形狀
        Dense(units=32, activation='relu'),                                    # 增加第一個隱藏層
        Dense(units=16, activation='relu'),                                    # 增加第二個隱藏層
        Dense(units=1)                                                         # 增加輸出層
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)                       # 設置 Adam 優化器
    model_NN2.compile(optimizer=adam, loss='mae')                              # 編譯模型
    
    # NN3 建模
    model_NN3 = Sequential([
        Input(shape=(length,)),                                                # 增加輸入層來指定輸入形狀
        Dense(units=32, activation='relu'),                                    # 增加第一個隱藏層
        Dense(units=16, activation='relu'),                                    # 增加第二個隱藏層
        Dense(units=8, activation='relu'),                                     # 增加第三個隱藏層
        Dense(units=1)                                                         # 增加輸出層
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)                       # 設置 Adam 優化器
    model_NN3.compile(optimizer=adam, loss='mae')                              # 編譯模型
    
    # NN4 建模
    model_NN4 = Sequential([
        Input(shape=(length,)),                                                # 增加輸入層來指定輸入形狀
        Dense(units=32, activation='relu'),                                    # 增加第一個隱藏層
        Dense(units=16, activation='relu'),                                    # 增加第二個隱藏層
        Dense(units=8, activation='relu'),                                     # 增加第三個隱藏層
        Dense(units=4, activation='relu'),                                     # 增加第四個隱藏層
        Dense(units=1)                                                         # 增加輸出層
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)                       # 設置 Adam 優化器
    model_NN4.compile(optimizer=adam, loss='mae')                              # 編譯模型
    
    # NN5 建模
    model_NN5 = Sequential([
        Input(shape=(length,)),                                                # 增加輸入層來指定輸入形狀
        Dense(units=32, activation='relu'),                                    # 增加第一個隱藏層
        Dense(units=16, activation='relu'),                                    # 增加第二個隱藏層
        Dense(units=8, activation='relu'),                                     # 增加第三個隱藏層
        Dense(units=4, activation='relu'),                                     # 增加第四個隱藏層
        Dense(units=2, activation='relu'),                                     # 增加第五個隱藏層
        Dense(units=1)                                                         # 增加輸出層
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)                       # 設置 Adam 優化器
    model_NN5.compile(optimizer=adam, loss='mae')                              # 編譯模型
    
    
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
        # mom 值
        X = df_X.iloc[:-i-2,:].values
        # mom IC 值
        Y = df_Y.iloc[1:-i-1,:].values
        
        # 設定 NN1 ~ NN5 訓練集 (待確認)
        # mom 值
        X_train = df_X.iloc[:-i-98,:].values
        # mom IC 值
        Y_train = df_Y.iloc[1:-i-97,:].values

        # 設定 NN1 ~ NN5 驗證集
        # mom 值   
        X_validation = df_X.iloc[-i-98:-2,:].values
        # mom IC 值
        Y_validation = df_Y.iloc[-i-97:-1,:].values
        
        # 設定測試集
        test_data = df_X.iloc[-i-2:-i-1,:].values


        ### 丟入數據訓練
        reg.fit(X,Y)
        regr.fit(X,Y.ravel())
        
        model_NN1.fit(
            X_train, Y_train,                                                  # 訓練數據和標籤
            validation_data=(X_validation, Y_validation),                      # 驗證數據和標籤
            batch_size=32,                                                     # 模型在每次更新權重時會使用 32 個樣本
            epochs=10                                                          # 訓練 10 個迭代
        )
        
        model_NN2.fit(
            X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            batch_size=32,
            epochs=10
        )
        
        model_NN3.fit(
            X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            batch_size=32,
            epochs=10
        )
        
        model_NN4.fit(
            X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            batch_size=32,
            epochs=10
        )
        
        model_NN5.fit(
            X_train, Y_train,
            validation_data=(X_validation, Y_validation),
            batch_size=32,
            epochs=10
        )

        
        # 預測結果
        pred = reg.predict(test_data)[0][0]
        predregr = regr.predict(test_data)[0]
        prednn1 = model_NN1.predict(test_data)[0][0].astype(np.float64)
        prednn2 = model_NN2.predict(test_data)[0][0].astype(np.float64)
        prednn3 = model_NN3.predict(test_data)[0][0].astype(np.float64)
        prednn4 = model_NN4.predict(test_data)[0][0].astype(np.float64)
        prednn5 = model_NN5.predict(test_data)[0][0].astype(np.float64)
        
        # 將結果加入到列表中
        predict_IC.append((year, month, pred, predregr, prednn1, prednn2, prednn3 ,prednn4, prednn5))



    # 將 results 轉換為 DataFrame
    df_predict_IC = pd.DataFrame(predict_IC, 
                                 columns=['year', 'month', 'Linear', 'RandomForest', 
                                          'NN1', 'NN2', 'NN3', 'NN4', 'NN5'])
    
    return df_predict_IC