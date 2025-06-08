import pandas as pd
import numpy as np


# %%  Import function

from fillna_utils import fillna_with_column_median


# %%  Function

def Information_Coefficient(df_price, df_log_return, h_period):
    
    log_return = df_log_return.copy()
    
    # 計算每個月每間公司的 h 個月動能
    if h_period == 1:
        momentum_h = np.log(df_price.shift(1, axis=1) / df_price.shift(2, axis=1))
    else:
        momentum_h = np.log(df_price.shift(2, axis=1) / df_price.shift(h_period + 1, axis=1))

    
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
    
    # 進行 Rank 排序 (出現小數點：有相同股價)
    log_return_rank = log_return.rank(axis=0, na_option='keep')
    momentum_h_rank = momentum_h.rank(axis=0, na_option='keep')
    
    # 計算 IC 值
    Rank_IC_list   = []
    Normal_IC_list = []
    
    for month in log_return.columns:
        
        # normal IC: Pearson correlation
        log_return_monthly = log_return[month]
        momentum_h_monthly = momentum_h[month]
        Normal_IC = log_return_monthly.corr(momentum_h_monthly)
        Normal_IC_list.append(Normal_IC)
        
        # rank IC: Spearman correlation (透過 rank + Pearson)
        log_return_rank_monthly = log_return_rank[month]
        momentum_h_rank_monthly = momentum_h_rank[month]
        Rank_IC = log_return_rank_monthly.corr(momentum_h_rank_monthly)
        Rank_IC_list.append(Rank_IC)

    # 整理成 DataFrame
    IC = pd.DataFrame({
        'Normal_IC': Normal_IC_list,
        'Rank_IC': Rank_IC_list
    }, index=log_return.columns)
    
    # 將動能資料轉置(Index = 時間，column = 股票代號)
    momentum_h      = momentum_h.T
    momentum_h_rank = momentum_h_rank.T.sort_index(ascending=False)
    
    return momentum_h, momentum_h_rank, IC
