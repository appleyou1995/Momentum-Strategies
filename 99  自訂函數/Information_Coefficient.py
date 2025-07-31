import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr


# %%  Import function

from fillna_utils import fillna_with_column_median


# %%  Function

def Information_Coefficient(df_price, h_period):
    
    # Calculate log return: R_{t} = ln(S_{t+1}/S_{t})
    log_return = np.log(df_price.shift(-1, axis=1) / df_price)
    
    ### 計算每個月每間公司的 h 個月動能
    # 滾動區間長度（取決於 h_period）
    window = h_period + 2
    
    # 建立完整資料遮罩：S_t ~ S_{t-1-h} 都要有值
    count_valid = df_price.rolling(window=window, axis=1).count()
    mask_valid = (count_valid == window)
    
    # 動能分子與分母
    if h_period == 1:
        numerator   = df_price.shift(1, axis=1)                                # S_{t-1}
        denominator = df_price.shift(2, axis=1)                                # S_{t-2}
    else:
        numerator   = df_price.shift(2, axis=1)                                # S_{t-2}
        denominator = df_price.shift(h_period + 1, axis=1)                     # S_{t-1-h}

    # 計算 momentum，缺資料則為 NaN
    momentum_h = np.where(mask_valid, np.log(numerator / denominator), np.nan)
    momentum_h = pd.DataFrame(momentum_h, index=df_price.index, columns=df_price.columns)

    
    # 比對 log_return 和 momentum_h 相同時間是否都有值，若其中一個沒有，則另一個也改為 NaN
    mask_nan = momentum_h.isna() | log_return.isna()
    momentum_h[mask_nan] = np.nan
    log_return[mask_nan] = np.nan

    # 刪除整個月份資料都是 NaN 的欄
    log_return.dropna(axis=1, how='all', inplace=True)
    momentum_h.dropna(axis=1, how='all', inplace=True)

    # 將空值填入當月的中位數
    log_return = fillna_with_column_median(log_return)
    momentum_h = fillna_with_column_median(momentum_h)
    
    # 轉換為浮點數格式
    log_return = log_return.apply(pd.to_numeric, errors='coerce')
    momentum_h = momentum_h.apply(pd.to_numeric, errors='coerce')
    
    ### 計算 IC 值
    Normal_IC_list = []
    Rank_IC_list   = []
    
    for month in log_return.columns:        
        # Pearson correlation (Normal IC)
        Normal_IC = pearsonr(log_return[month], momentum_h[month])[0]
        Normal_IC_list.append(Normal_IC)
    
        # Spearman correlation (Rank IC)
        Rank_IC = spearmanr(log_return[month], momentum_h[month])[0]
        Rank_IC_list.append(Rank_IC)

    # 整理成 DataFrame
    IC = pd.DataFrame({
        'Normal_IC': Normal_IC_list,
        'Rank_IC': Rank_IC_list
    }, index=log_return.columns)
    
    # 將動能資料轉置(Index = 時間，column = 股票代號)
    momentum_h = momentum_h.T
    
    return momentum_h, IC
