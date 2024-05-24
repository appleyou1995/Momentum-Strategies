import pandas as pd
import numpy as np


# %%  Import function

from fillna_with_mean import fillna_with_column_mean


# %%

def Information_Coefficient(df_price, df_log_return, h_period):
    
    log_return = df_log_return.copy()
    
    # 計算每個月每間公司的一個月動能
    momentum_h = np.log(df_price/df_price.shift(h_period, axis=1))
    
    # 比對 log_return 和 momentum_h 相同時間是否都有值，若其中一個沒有，則另一個也改為 NaN
    for i in range(momentum_h.shape[0]):
        for j in range(momentum_h.shape[1]):
            if pd.isna(momentum_h.iat[i, j]) or pd.isna(log_return.iat[i, j]):
                    momentum_h.iat[i, j] = np.nan
                    log_return.iat[i, j] = np.nan

    # 刪除整個月份資料都是 NaN 的欄
    log_return.dropna(axis=1, how='all', inplace=True)
    momentum_h.dropna(axis=1, how='all', inplace=True)

    # 將空值填入當月的平均值
    log_return = fillna_with_column_mean(log_return)
    momentum_h = fillna_with_column_mean(momentum_h)    
    
    # 轉換為浮點數格式
    log_return = log_return.apply(pd.to_numeric, errors='coerce')
    momentum_h = momentum_h.apply(pd.to_numeric, errors='coerce')
    
    # 進行 Rank 排序 (出現小數點：有相同股價)
    log_return_rank = log_return.rank(axis=0, na_option='keep')
    momentum_h_rank = momentum_h.rank(axis=0, na_option='keep')
    
    # 計算 IC 值
    corr_list = []
    
    for month in log_return_rank.columns:
        log_return_monthly = log_return_rank[month]
        momentum_h_monthly = momentum_h_rank[month]
        corr = log_return_monthly.corr(momentum_h_monthly)
        corr_list.append(corr)

    IC = pd.DataFrame({'IC': corr_list}, index=log_return_rank.columns)
    
    # 將動能資料轉置(Index = 時間，column = 股票代號)
    momentum_h = momentum_h.T
    
    return momentum_h, IC
