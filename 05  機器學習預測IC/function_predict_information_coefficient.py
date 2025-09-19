import os, warnings
# import gc

# ===== 設環境變數 =====
seed = 999
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 只顯示 ERROR，其他 INFO、WARNING 不顯示
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制用 CPU 訓練，確保隨機種子固定
os.environ['PYTHONHASHSEED'] = str(seed)   # 固定 Python hash 隨機性

# ===== imports =====
import numpy      as np
import pandas     as pd
import tensorflow as tf

# ========= sklearn =========
# from sklearn.linear_model        import LinearRegression, HuberRegressor, SGDRegressor
# from sklearn.ensemble            import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.decomposition       import PCA
# from sklearn.pipeline            import make_pipeline
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.preprocessing       import SplineTransformer
# from sklearn.metrics             import mean_squared_error
from sklearn.exceptions          import ConvergenceWarning

# ========= keras =========
# from tensorflow.keras.layers     import Dense, Input
# from tensorflow.keras.models     import Sequential
# from tensorflow.keras.callbacks  import EarlyStopping

# ========= cvxpy（精確的 group-lasso + Huber）=========
# import cvxpy as cp

# ===== 設定隨機種子 =====
tf.keras.utils.set_random_seed(seed)       # 同時固定 random、numpy、tensorflow 的隨機性

# ===== 隱藏 warning =====
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # 關掉 HuberRegressor 收斂警告
warnings.filterwarnings(
    "ignore",
    message=r"Objective contains too many subexpressions.*",    # GLM+H：關掉 CVXPY 的提示
    category=UserWarning
)
tf.get_logger().setLevel("ERROR") # NN1–NN5：把所有 INFO 和 WARNING 訊息關掉，只留下錯誤


# %%  Utilities

def fillna_by_row_median_inplace(X):
    # 以列中位數填 NaN；回傳新陣列（float32）
    X = X.astype(np.float32, copy=True)
    med = np.nanmedian(X, axis=1)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[0]).astype(np.float32)
    return X


# %%  Models

from model_ols        import model_OLS
from model_ols_huber  import model_OLS_Huber
from model_enet_huber import model_ENet_Huber
from model_pcr        import model_PCR
from model_pls        import model_PLS
from model_glm_huber  import model_GLM_Huber
from model_gbrt_huber import model_GBRT_Huber
from model_rf         import model_RF
from model_nn         import model_NN, layers_dict, early_stopping


# # %%  2-1) OLS

# def model_OLS(X, Y, X_test):
#     reg = LinearRegression().fit(X, Y.ravel())
#     return float(reg.predict(X_test)[0])


# # %%  2-2) OLS+H

# def model_OLS_Huber(X, Y, X_test):
#     m = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=1000)
#     m.fit(X, Y.ravel())
#     return float(m.predict(X_test)[0])


# # %%  2-3) ENet+H

# def model_ENet_Huber(X, Y, X_test, seed=999):
#     m = SGDRegressor(loss='huber', penalty='elasticnet',
#                      alpha=3e-3, l1_ratio=0.9, epsilon=0.05,
#                      max_iter=3000, tol=1e-4, random_state=seed)
#     m.fit(X, Y.ravel())
#     return float(m.predict(X_test)[0])


# # %%  2-4) PCR

# def model_PCR(X, Y, X_test, K_max=20):
#     K = int(min(K_max, X.shape[1], max(1, X.shape[0]-1)))
#     pipe = make_pipeline(PCA(n_components=K), LinearRegression())
#     pipe.fit(X, Y.ravel())
#     return float(pipe.predict(X_test)[0])


# # %%  2-5) PLS

# def model_PLS(X, Y, X_test, K=3):
#     k_use = int(min(K, X.shape[1], max(1, X.shape[0]-1)))
#     m = PLSRegression(n_components=k_use).fit(X, Y.ravel())
#     return float(m.predict(X_test)[0])


# # %%  2-6) GLM+H

# def choose_solver():
#     # 選擇已安裝的解器，依偏好順序回傳
#     installed = set(cp.installed_solvers())
#     for s in ("OSQP", "SCS", "ECOS"):
#         if s in installed:
#             return s
#     return None

# # GLM + H (Huber) with Group-Lasso via CVXPY
# # - group = 同一個「原始特徵」展開後的所有樣條基底

# def spline_design_and_groups(X_raw, degree=3, n_knots=5):
#     # 將原始 X 做 Spline 展開，並回傳 groups 向量（以原始特徵為組）。
#     spl = SplineTransformer(degree=degree, n_knots=n_knots, include_bias=False)
#     Xs = spl.fit_transform(X_raw)   # (n, p_spline)
#     n_feat = X_raw.shape[1]
#     width = Xs.shape[1] // n_feat   # 每個特徵展開的基底數
#     groups = np.repeat(np.arange(n_feat), width)
#     return Xs.astype(np.float32), groups.astype(int), spl

# def fit_glm_group_huber_predict(X_tr, y_tr, X_va, y_va, X_te,
#                                 n_knots_grid=(5,), degree=3,
#                                 lam_grid=(1e-3,), delta=1.35):
#     # 以驗證集挑選 (n_knots, lambda)，精確的 Huber + Group Lasso。
#     # 回傳 X_te 的單點預測值（float）。
#     best_loss, best_beta, best_spl = np.inf, None, None
#     solver = choose_solver()

#     for n_knots in n_knots_grid:
#         # Spline 展開（訓練用的 spl 也拿來轉換 valid/test）
#         Xs_tr, groups, spl = spline_design_and_groups(X_tr, degree=degree, n_knots=n_knots)
#         Xs_va = spl.transform(X_va).astype(np.float32)
#         Xs_te = spl.transform(X_te).astype(np.float32)

#         n, p = Xs_tr.shape
#         # 為了公平，對每組標準化（避免某組變數尺度較大吞掉懲罰）
#         col_std = Xs_tr.std(axis=0, ddof=0)
#         col_std[col_std == 0] = 1.0
#         Xs_tr_std = Xs_tr / col_std
#         Xs_va_std = Xs_va / col_std
#         Xs_te_std = Xs_te / col_std

#         # 權重 w_g = sqrt(p_g)（避免大組吃虧）
#         w_g = {}
#         for g in np.unique(groups):
#             p_g = (groups == g).sum()
#             w_g[g] = np.sqrt(p_g)

#         for lam in lam_grid:
#             beta = cp.Variable(p)
#             # Huber 損失
#             loss = cp.sum(cp.huber(y_tr - Xs_tr_std @ beta, delta))
#             # Group-Lasso 懲罰
#             pen = 0
#             for g in np.unique(groups):
#                 idx = np.where(groups == g)[0]
#                 pen += w_g[g] * cp.norm2(beta[idx])
#             obj = cp.Minimize(loss + lam * pen)
#             prob = cp.Problem(obj)
#             try:
#                 prob.solve(solver=solver, verbose=False)
#             except Exception:
#                 # 換 solver 再試（以防某些環境某解器不可用）
#                 for s in ["SCS", "ECOS", "OSQP"]:
#                     try:
#                         prob.solve(solver=s, verbose=False)
#                         break
#                     except Exception:
#                         continue

#             if beta.value is None:
#                 continue

#             # 驗證集損失（用 MSE 或 Huber 皆可；用 MSE 比較直觀）
#             y_hat_va = (Xs_va_std @ beta.value).ravel()
#             val_loss = mean_squared_error(y_va, y_hat_va)

#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 best_beta = beta.value.copy()
#                 best_spl  = spl
#                 best_std  = col_std.copy()

#     # 以最佳參數做測試點預測
#     Xs_te = best_spl.transform(X_te).astype(np.float32)
#     Xs_te_std = Xs_te / best_std
#     y_hat_te = float((Xs_te_std @ best_beta).ravel()[0])
#     return y_hat_te

# def model_GLM_Huber(X, Y, X_valid, Y_valid, X_test):
#     return float(fit_glm_group_huber_predict(
#         X_tr=X, y_tr=Y.ravel(),
#         X_va=X_valid, y_va=Y_valid.ravel(),
#         X_te=X_test,
#         n_knots_grid=(5,), degree=3,
#         lam_grid=(1e-3,), delta=1.35
#     ))


# # %%  2-7) GBRT+H

# def model_GBRT_Huber(X, Y, X_test, seed=999):
#     m = GradientBoostingRegressor(
#         loss='huber', max_depth=2,
#         learning_rate=0.1, n_estimators=100,
#         random_state=seed
#     ).fit(X, Y.ravel())
#     return float(m.predict(X_test)[0])


# # %%  2-8) RF

# def model_RF(X, Y, X_test, seed=999):
#     m = RandomForestRegressor(
#         max_depth=3, n_estimators=100,
#         random_state=seed, n_jobs=-1
#     ).fit(X, Y.ravel())
#     return float(m.predict(X_test)[0])


# # %%  2-9) NN1 - NN5

# early_stopping = EarlyStopping(
#     monitor='val_loss',        # 要監控的指標：驗證集損失（val_loss）
#     patience=5,                # 連續 5 個 epoch 沒「明顯改善」就提前停止（patience 要比總 epoch 小才有意義）
#     min_delta=1e-4,            # 視為「有改善」的最小幅度，避免極小波動被當成進步
#     mode='min',                # 該指標越小越好（loss 適用 'min'）
#     restore_best_weights=True, # 停止時把模型權重回復到 val_loss 最佳的那次
#     verbose=0                  # 設為 1 可在訓練過程印出 EarlyStopping 的訊息
# )

# def build_NN_model(n_features, layers):
#     model = Sequential()
#     model.add(Input(shape=(n_features,)))
#     for units in layers:
#         model.add(Dense(units, activation='relu'))
#     model.add(Dense(units=1))
#     adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # learning_rate 更新幅度小，收斂較慢，但較穩定
#     model.compile(optimizer=adam, 
#                   loss='mse')
#     return model

# def model_NN(X_train, Y_train, X_valid, Y_valid, X_test, layers, patience_cb, seed=999):
#     tf.keras.backend.clear_session()
#     gc.collect()
#     n_features = X_train.shape[1]
#     m = build_NN_model(n_features, layers)
#     m.fit(X_train, Y_train,
#           validation_data=(X_valid, Y_valid),
#           epochs=50, batch_size=64, verbose=0, shuffle=False,
#           callbacks=[patience_cb])
#     return float(m.predict(X_test, verbose=0)[0][0])


# %%  Main function

def predict_information_coefficient(
        df_X,               # Momentum:   index=YYYY-MM（月）, columns=PERMNO...；值可含 NaN
        df_Y,               # IC:         index=YYYY-MM（月）, 單欄或 shape (*,1)
        test_start_date,    # 'YYYY-MM'：第一個要預測的 Y 的月份
        valid_length,       # 驗證集長度（僅在需要驗證集的模型時才會用到）
        models_to_run,      # e.g. ['OLS','ENet+H','PCR']，也支援 'NN'（同義於 NN1~NN5 全跑）
        output_dir,         # e.g. Path_Output
        h,                  # 檔名中的 {h}，例如 horizon 或你的既有變數
        verbose=True):
        
    
    # --- 尋找測試集起訖月份的位置 ---
    date_list = df_Y.index.tolist()
    test_start_pos = date_list.index(test_start_date)
    test_end_pos   = len(date_list) - 1

    # --- numpy 版本 ---
    X_all = df_X.values.astype(np.float32)
    Y_all = df_Y.values.astype(np.float32)    
    
    # --- 建資料夾 ---
    os.makedirs(output_dir, exist_ok=True)    
    
    # --- 結果收集（整合在一起，之後再拆檔存） ---
    all_rows = []    

    # --- 將 'NN' 展開為 NN1~NN5 ---
    # layers_dict = {
    #     'NN1': [32],
    #     'NN2': [32, 16],
    #     'NN3': [32, 16, 8],
    #     'NN4': [32, 16, 8, 4],
    #     'NN5': [32, 16, 8, 4, 2],
    # }
    expanded = []
    for m in models_to_run:
        if m.upper() == 'NN':
            expanded.extend(['NN1','NN2','NN3','NN4','NN5'])
        else:
            expanded.append(m)
    models_to_run = expanded
    
    # --- 哪些模型需要驗證集 ---
    MODELS_NEED_VALID = set(['GLM+H', 'NN1','NN2','NN3','NN4','NN5'])
    
    # --- 是否需要驗證集（若指定模型裡包含需要驗證的） ---
    need_valid = any(m in MODELS_NEED_VALID for m in models_to_run)
    
    
    for n in range(test_start_pos, test_end_pos + 1):
        
        current_date = date_list[n]

        if verbose:
            print(f"h = {h}, n = {n}, date = {current_date}")
        
        ################## 樣本切割 ##################
        
        # ------- 測試期的可交易標的 -------
        X_test_row = X_all[n-1, :]
        tradable_mask = ~np.isnan(X_test_row)
        X_test = X_test_row[tradable_mask][None, :]
        
        # ------- 訓練資料（給所有非 NN / 不驗證 的模型）-------
        X = X_all[0:n-1, :][:, tradable_mask]
        Y = Y_all[1:n, 0]
        X = fillna_by_row_median_inplace(X)
        
        # ------- 若任何模型需要驗證集，才切 & 準備 NN 用資料 -------
        if need_valid:
            X_train = X_all[0:n-1-valid_length, :][:, tradable_mask]
            Y_train = Y_all[1:n-valid_length, 0]
            X_train = fillna_by_row_median_inplace(X_train)

            X_valid = X_all[n-1-valid_length:n-1, :][:, tradable_mask]
            Y_valid = Y_all[n-valid_length:n, 0]
            X_valid = fillna_by_row_median_inplace(X_valid)
        else:
            X_train = Y_train = X_valid = Y_valid = None  # 供呼叫端判斷（不會用到）
        
        # ------- dateframe check -------
        # X_test_all = df_X.iloc[n-1:n, :]
        # X_test_clean = X_test_all.dropna(axis=1, how="all")
        # tradable = X_test_clean.columns.tolist()   # 篩選出當月有值的可以交易股票
        # X_test = X_test_clean.values.astype(np.float32)
        # Y = df_Y.iloc[1:n, :].values.astype(np.float32)
        # X = df_X.iloc[0:n-1, :][tradable]
        # X = fillna_by_row_median(X).values.astype(np.float32)        
        # Y_train = df_Y.iloc[1:n-valid_length, :].values.astype(np.float32)
        # X_train = df_X.iloc[0:n-1-valid_length, :][tradable]
        # X_train = fillna_by_row_median(X_train).values.astype(np.float32)
        # Y_valid = df_Y.iloc[n-valid_length:n, :].values.astype(np.float32)
        # X_valid = df_X.iloc[n-1-valid_length:n-1, :][tradable]
        # X_valid = fillna_by_row_median(X_valid).values.astype(np.float32)
        
                
        ################## 模型預測 ##################
        
        # ------- 初始化一列結果 -------
        row = {'date': current_date}
        
        # ------- 逐模型呼叫對應函數 -------
        for model_name in models_to_run:
            # print(f"    {model_name}")                

            if model_name == 'OLS':
                row['OLS'] = model_OLS(X, Y, X_test)

            elif model_name == 'OLS+H':
                row['OLS+H'] = model_OLS_Huber(X, Y, X_test)

            elif model_name == 'ENet+H':
                row['ENet+H'] = model_ENet_Huber(X, Y, X_test, seed=seed)

            elif model_name == 'PCR':
                row['PCR'] = model_PCR(X, Y, X_test, K_max=20)

            elif model_name == 'PLS':
                row['PLS'] = model_PLS(X, Y, X_test, K=3)

            elif model_name == 'GLM+H':
                if not need_valid:
                    raise RuntimeError("GLM+H 需要驗證集，但目前未切驗證集。")
                row['GLM+H'] = model_GLM_Huber(X, Y, X_valid, Y_valid, X_test)

            elif model_name == 'GBRT+H':
                row['GBRT+H'] = model_GBRT_Huber(X, Y, X_test, seed=seed)

            elif model_name == 'RF':
                row['RF'] = model_RF(X, Y, X_test, seed=seed)

            elif model_name in layers_dict:  # NN1..NN5
                if not need_valid:
                    raise RuntimeError(f"{model_name} 需要驗證集，但目前未切驗證集。")
                row[model_name] = model_NN(
                    X_train, Y_train, X_valid, Y_valid, X_test,
                    layers=layers_dict[model_name],
                    patience_cb=early_stopping,
                    seed=seed
                )

            else:
                raise ValueError(f"未知的模型名稱：{model_name}")

        all_rows.append(row)
    
    
    # --- 合併 DataFrame（只含 date + 選定模型欄） ---
    cols = ['date'] + models_to_run
    df_pred_all = pd.DataFrame(all_rows, columns=cols)

    # --- 依模型各自輸出 CSV ---
    for model_name in models_to_run:
        df_one = df_pred_all[['date', model_name]].rename(columns={model_name: 'pred'})
        out_path = os.path.join(output_dir, f'predict_Normal_IC_{h}_{model_name}.csv')
        df_one.to_csv(out_path, index=False)
        if verbose:
            print(f'[output] predict_Normal_IC_{h}_{model_name}.csv')

    return df_pred_all
