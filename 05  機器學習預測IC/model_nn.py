import gc
import tensorflow as tf
from tensorflow.keras.layers    import Dense, Input
from tensorflow.keras.models    import Sequential
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(
    monitor='val_loss',        # 要監控的指標：驗證集損失（val_loss）
    patience=5,                # 連續 5 個 epoch 沒「明顯改善」就提前停止（patience 要比總 epoch 小才有意義）
    min_delta=1e-4,            # 視為「有改善」的最小幅度，避免極小波動被當成進步
    mode='min',                # 該指標越小越好（loss 適用 'min'）
    restore_best_weights=True, # 停止時把模型權重回復到 val_loss 最佳的那次
    verbose=0                  # 設為 1 可在訓練過程印出 EarlyStopping 的訊息
)

layers_dict = {
    'NN1': [32],
    'NN2': [32, 16],
    'NN3': [32, 16, 8],
    'NN4': [32, 16, 8, 4],
    'NN5': [32, 16, 8, 4, 2],
}

def build_NN_model(n_features, layers):
    model = Sequential()
    model.add(Input(shape=(n_features,)))
    for units in layers:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(units=1))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mse')
    return model

def model_NN(X_train, Y_train, X_valid, Y_valid, X_test, layers, patience_cb, seed=999):
    tf.keras.backend.clear_session()
    gc.collect()
    n_features = X_train.shape[1]
    m = build_NN_model(n_features, layers)
    m.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=50, batch_size=64, verbose=0, shuffle=False,
        callbacks=[patience_cb]
    )
    return float(m.predict(X_test, verbose=0)[0][0])
