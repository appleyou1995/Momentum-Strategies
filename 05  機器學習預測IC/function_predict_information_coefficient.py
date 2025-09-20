import os, random, warnings

# ===== 設環境變數 =====
seed = 999
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 只顯示 ERROR，其他 INFO、WARNING 不顯示
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制用 CPU 訓練，確保隨機種子固定
os.environ['PYTHONHASHSEED'] = str(seed)   # 固定 Python hash 隨機性

# ===== imports =====
import numpy      as np
import pandas     as pd
import tensorflow as tf

# ========= sklearn =========
from sklearn.exceptions import ConvergenceWarning

# ========= random seed =========
tf.keras.utils.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ========= warning =========
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # 關掉 HuberRegressor 收斂警告
warnings.filterwarnings(
    "ignore",
    message=r"Objective contains too many subexpressions.*",    # GLM+H：關掉 CVXPY 的提示
    category=UserWarning
)
tf.get_logger().setLevel("ERROR") # NN1–NN5：把所有 INFO 和 WARNING 訊息關掉，只留下錯誤


# %%  Utility

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
from model_nn         import model_NN, layers_dict


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
    expanded = []
    for m in models_to_run:
        if m.upper() == 'NN':
            expanded.extend(['NN1','NN2','NN3','NN4','NN5'])
        else:
            expanded.append(m)
    models_to_run = expanded
    
    # --- 需要驗證集的模型 ---
    MODELS_NEED_VALID = set(['GLM+H', 'NN1','NN2','NN3','NN4','NN5'])
    need_valid = any(m in MODELS_NEED_VALID for m in models_to_run)
    
    
    for n in range(test_start_pos, test_end_pos + 1):
        
        current_date = date_list[n]

        if verbose:
            print(f"h = {h}, n = {n}, date = {current_date}")
        
        ################## 樣本切割 ##################
        
        # ------- 測試集與當期可交易標的 -------
        X_test_row = X_all[n-1, :]
        tradable_mask = ~np.isnan(X_test_row)
        X_test = X_test_row[tradable_mask][None, :]
        
        # ------- 訓練集（給所有非 NN / 不驗證 的模型）-------
        X = X_all[0:n-1, :][:, tradable_mask]
        Y = Y_all[1:n, 0]
        X = fillna_by_row_median_inplace(X)
        
        # ------- 驗證集 -------
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
                row['GLM+H'] = model_GLM_Huber(X, Y, X_valid, Y_valid, X_test)

            elif model_name == 'GBRT+H':
                row['GBRT+H'] = model_GBRT_Huber(X, Y, X_test, seed=seed)

            elif model_name == 'RF':
                row['RF'] = model_RF(X, Y, X_test, seed=seed)

            elif model_name in layers_dict:  # NN1...NN5
                row[model_name] = model_NN(
                    X_train, Y_train, X_valid, Y_valid, X_test,
                    layers=layers_dict[model_name],
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
