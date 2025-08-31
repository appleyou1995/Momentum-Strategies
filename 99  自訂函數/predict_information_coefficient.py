import os
import gc

# ===== 設環境變數 =====
seed = 999
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 只顯示 ERROR，其他 INFO、WARNING 不顯示
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制用 CPU 訓練，確保隨機種子固定
os.environ['PYTHONHASHSEED'] = str(seed)   # 固定 Python hash 隨機性

# ===== import 需要的套件 =====
import numpy      as np
import pandas     as pd
import tensorflow as tf
import warnings

from sklearn.ensemble           import RandomForestRegressor
from sklearn.linear_model       import LinearRegression
from tensorflow.keras.layers    import Dense, Input
from tensorflow.keras.models    import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# ===== 設定隨機種子 =====
tf.keras.utils.set_random_seed(seed)       # 同時固定 random、numpy、tensorflow 的隨機性

# ===== 隱藏 warning =====
warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")


# %%  Fill NaN by median

def fillna_by_row_median(df):
    med = df.median(axis=1)
    return df.T.fillna(med).T

        
# %%  NN model generator

early_stopping = EarlyStopping(
    monitor='val_loss',        # 要監控的指標：驗證集損失（val_loss）
    patience=5,                # 連續 5 個 epoch 沒「明顯改善」就提前停止（patience 要比總 epoch 小才有意義）
    min_delta=1e-4,            # 視為「有改善」的最小幅度，避免極小波動被當成進步
    mode='min',                # 該指標越小越好（loss 適用 'min'）
    restore_best_weights=True, # 停止時把模型權重回復到 val_loss 最佳的那次
    verbose=0                  # 設為 1 可在訓練過程印出 EarlyStopping 的訊息
)

def build_NN_model(n_features, layers):
    model = Sequential()
    model.add(Input(shape=(n_features,)))
    for units in layers:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(units=1))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, 
                  loss='mse')
    return model


# %%  Function

def predict_information_coefficient(
        df_X,               # Momentum
        df_Y,               # IC
        test_start_date,    # 字串 'YYYY-MM'，表示第一個要預測的 Y 的月份
        valid_length, 
        verbose=True):
    
    # 最後一個要預測的 Y 的月份
    test_end_date = df_Y.index[-1]
    
    # 尋找測試集起訖月份的位置
    date_list = df_Y.index.tolist()
    test_start_pos = date_list.index(test_start_date)
    test_end_pos   = date_list.index(test_end_date)    

    # 結果列表
    predict_IC = []
    
    for n in range(test_start_pos, test_end_pos + 1):
        
        current_date = date_list[n]

        if verbose:
            print(f"Current iteration: n = {n}, date = {current_date}")        
        
        # 清掉上一次的 TensorFlow graph，防止 memory 累積
        tf.keras.backend.clear_session()
        gc.collect()
        
        ### 樣本切割        
        # 測試集 (要預測的當期 IC)
        X_test_all = df_X.iloc[n-1:n, :]
        X_test_clean = X_test_all.dropna(axis=1, how="all")
        tradable = X_test_clean.columns.tolist()   # 篩選出當月有值的可以交易股票
        X_test = X_test_clean.values.astype(np.float32)
        
        # OLS & RF 訓練集
        Y = df_Y.iloc[1:n, :].values.astype(np.float32)
        X = df_X.iloc[0:n-1, :][tradable]
        X = fillna_by_row_median(X).values.astype(np.float32)
        
        # NN 訓練集
        Y_train = df_Y.iloc[1:n-valid_length, :].values.astype(np.float32)
        X_train = df_X.iloc[0:n-1-valid_length, :][tradable]
        X_train = fillna_by_row_median(X_train).values.astype(np.float32)
        
        # NN 驗證集
        Y_valid = df_Y.iloc[n-valid_length:n, :].values.astype(np.float32)
        X_valid = df_X.iloc[n-1-valid_length:n-1, :][tradable]
        X_valid = fillna_by_row_median(X_valid).values.astype(np.float32)
        
        ### 初始化模型
        
        # 線性模型
        reg_OLS = LinearRegression()
        
        # 隨機森林
        reg_RF = RandomForestRegressor(
            max_depth=3,        # 限制每棵樹的最大深度
            random_state=seed,  # 固定隨機種子，讓結果可重現
            n_estimators=100,   # 森林中樹的數量
            n_jobs=-1           # 使用所有 CPU 核心並行運算
        )
        
        # 類神經網路
        n_features = len(tradable)
        model_NN1 = build_NN_model(n_features, [32])
        model_NN2 = build_NN_model(n_features, [32,16])
        model_NN3 = build_NN_model(n_features, [32,16,8])
        model_NN4 = build_NN_model(n_features, [32,16,8,4])
        model_NN5 = build_NN_model(n_features, [32,16,8,4,2])
        
        ### 模型訓練
        reg_OLS.fit(X,Y.ravel())
        reg_RF.fit(X,Y.ravel())
        
        for m in (model_NN1, model_NN2, model_NN3, model_NN4, model_NN5):
            m.fit(X_train, Y_train,
                  validation_data=(X_valid, Y_valid),
                  epochs=100, batch_size=32, verbose=0,
                  shuffle=False, # 關閉洗牌，讓結果完全可重現
                  callbacks=[early_stopping])
        
        ### 預測結果
        pred_OLS = float(reg_OLS.predict(X_test)[0])
        pred_RF  = float(reg_RF.predict(X_test)[0])
        preds_NN = [float(m.predict(X_test, verbose=0)[0][0])
                    for m in (model_NN1, model_NN2, model_NN3, model_NN4, model_NN5)]
        
        ### 加入結果
        predict_IC.append((current_date, pred_OLS, pred_RF, *preds_NN))
    
    # 輸出 DataFrame
    df_predict_IC = pd.DataFrame(predict_IC, 
                                 columns=['date', 'OLS', 'RF', 
                                          'NN1', 'NN2', 'NN3', 'NN4', 'NN5'])
    
    return df_predict_IC
