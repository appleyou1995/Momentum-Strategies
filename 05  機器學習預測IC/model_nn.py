import os, gc
import numpy as np

os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
from sklearn.preprocessing      import StandardScaler
from tensorflow.keras           import regularizers
from tensorflow.keras.layers    import Dense, Input, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models    import Sequential
from tensorflow.keras.callbacks import EarlyStopping

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ========== Hyperparameters ==========

HP = {
    "LR":            5e-4,     # learning rate
    "L2":            1e-3,     # kernel L2 regularization
    "DROPOUT":       0.15,     # Dropout rate; <=0 means disabled
    "BATCH_SIZE":    64,       # mini-batch size
    "USE_BN":        True,     # whether to use Batch Normalization
    "MAX_EPOCHS":    50,       # maximum number of training epochs
    "PATIENCE":      5,        # epochs with no improvement before early stopping
    "MIN_DELTA":     1e-4,     # minimum change in monitored metric to qualify as improvement
    "AMSGRAD":       True,     # whether to use the AMSGrad variant of Adam
    "NEGATIVE_SLOPE": 0.05,    # LeakyReLU parameter (negative slope)
}


# For the record

def format_hp(hp: dict) -> str:
    parts = []
    for k, v in hp.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.1e}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)

# print(format_hp(HP))
# print(f"[h = {h}][{(r2_oos_no_demean(y_true, y_pred) * 100).round(4)}] {format_hp(HP)}")


# ------------------------------------------------

def report_history(history, es_cb=None, key='val_loss'):

    loss_arr     = history.history['loss']
    val_loss_arr = history.history[key]

    best_idx   = int(np.argmin(val_loss_arr))
    best_epoch = best_idx + 1
    best_loss  = loss_arr[best_idx]
    best_vloss = val_loss_arr[best_idx]
    print(f"    Best:  epoch={best_epoch}, loss={best_loss:.4f}, val_loss={best_vloss:.4f}")

    # last_epoch = len(loss_arr)
    # last_loss  = loss_arr[-1]
    # last_vloss = val_loss_arr[-1]
    # print(f"    Last:  epoch={last_epoch}, loss={last_loss:.4f}, val_loss={last_vloss:.4f}")

    # stopped_epoch = None
    # if es_cb is not None and hasattr(es_cb, "stopped_epoch"):
    #     stopped_epoch = es_cb.stopped_epoch or None
    # if stopped_epoch is not None:
    #     print(f"    Early stopping at epoch: {stopped_epoch}")
    # else:
    #     print("    Early stopping not triggered")
    
    print(" ")


# ------------------------------------------------

layers_dict = {
    'NN1': [32],
    'NN2': [32, 16],
    'NN3': [32, 16, 8],
    'NN4': [32, 16, 8, 4],
    'NN5': [32, 16, 8, 4, 2],
}

def build_NN_model(n_features, layers, hp: dict):
    
    use_bn = bool(hp["USE_BN"])
    use_dropout = float(hp["DROPOUT"]) > 0.0
    p_drop = float(hp["DROPOUT"])
    
    model = Sequential()
    model.add(Input(shape=(n_features,)))
    for units in layers:
        # Strong regularization
        model.add(Dense(units, 
                        kernel_regularizer=regularizers.l2(hp["L2"])))
        if use_bn:
            model.add(BatchNormalization())
        model.add(LeakyReLU(negative_slope=hp["NEGATIVE_SLOPE"]))
        if use_dropout:
            model.add(Dropout(p_drop))
    model.add(Dense(1))
    adam = tf.keras.optimizers.Adam(learning_rate=hp["LR"], 
                                    amsgrad=hp["AMSGRAD"])
    model.compile(optimizer=adam, loss='mse')
    return model


def model_NN(X_train, Y_train,
             X_valid, Y_valid, 
             X_test, 
             layers, seed=999, hp: dict = HP):
    
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    
    # X: feature-wise standardization (fit on training)
    x_scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = x_scaler.fit_transform(X_train)
    X_valid_s = x_scaler.transform(X_valid)
    X_test_s  = x_scaler.transform(X_test)
    
    # Y: standardize & later invert back
    y_scaler = StandardScaler()
    Y_train_s = y_scaler.fit_transform(Y_train.reshape(-1, 1)).ravel()
    Y_valid_s = y_scaler.transform(Y_valid.reshape(-1, 1)).ravel()
    
    gc.collect()
    n_features = X_train_s.shape[1]
    
    m = build_NN_model(n_features, layers, hp)
    
    patience_cb = EarlyStopping(
        monitor='val_loss',
        patience=hp["PATIENCE"],
        min_delta=hp["MIN_DELTA"],
        mode='min',
        restore_best_weights=True,
        verbose=0
    )
    
    history = m.fit(
        X_train_s, Y_train_s,
        validation_data=(X_valid_s, Y_valid_s),
        epochs=hp["MAX_EPOCHS"],
        batch_size=hp["BATCH_SIZE"],
        verbose=0,
        shuffle=False,
        callbacks=[patience_cb]
    )
    
    y_pred_s = m.predict(X_test_s, verbose=0)[0][0]
    y_pred   = y_scaler.inverse_transform([[y_pred_s]])[0][0]
    
    report_history(history, es_cb=patience_cb)
    
    return float(y_pred)
