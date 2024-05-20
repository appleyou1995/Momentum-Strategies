import os
import sys
import pandas as pd
import numpy as np


# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)


# %%  Mac 資料夾路徑

Path_Mac = '/Users/irisyu/Library/CloudStorage/GoogleDrive-jouping.yu@gmail.com/'
Path_dir = os.path.join(Path_Mac, Path_PaperFolder)


# %%  Input and Output Path

Path_Input  = os.path.join(Path_dir, 'Data/')
Path_Output = os.path.join(Path_dir, 'Data/')


# %%  Import data

PRC         = pd.read_csv(os.path.join(Path_Input, 'PRC.csv')        , index_col='PERMNO')
log_returns = pd.read_csv(os.path.join(Path_Input, 'log_returns.csv'), index_col='PERMNO')
log_returns1 = log_returns


# %%  計算每個月每間公司的一個月動能(從1995-02開始有值)(ln(t/t-1))

mom = np.log( PRC / PRC.shift(1, axis=1))
mom1 = mom


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from fillna_with_row_mean import fillna_with_row_mean


# %%  處理空值

# 比對 log_returns1 和 mom1 相同時間是否都有值，若其中一個沒有，則另一個也改為 NaN
for i in range(mom1.shape[0]):
    for j in range(mom1.shape[1]):
        if pd.isna(mom1.iat[i, j]) or pd.isna(log_returns1.iat[i, j]):
                mom1.iat[i, j] = np.nan
                log_returns1.iat[i, j] = np.nan


# 刪除整個月份資料都是 NaN 的欄(頭尾月份)
log_returns1_new = log_returns1.copy()
log_returns1_new.dropna(axis=1, how='all', inplace=True)
mom1_new = mom1.copy()
mom1_new.dropna(axis=1, how='all', inplace=True)


# 將 log_returns1_new 和 mom1_new 中的空值填入各自的平均值
log_returns1_new = log_returns1_new.apply(fillna_with_row_mean, axis=1)
mom1_new = mom1_new.apply(fillna_with_row_mean, axis=1)


# %%
## 將 log_returns_new 和 mom1_new 轉換為浮點數格式
log_returns1_new = log_returns1_new.apply(pd.to_numeric, errors='coerce')
mom1_new = mom1_new.apply(pd.to_numeric, errors='coerce')


# 測試排序用
# test1 = log_returns1_new.iloc[:5,:5]
# test1_rank = test1.rank(axis=1, na_option='keep')



# 對 log_returns1_new 和 mom1_new 分別進行 Rank 排序
log_returns1_rank = log_returns1_new.copy()
log_returns1_rank = log_returns1_rank.rank(axis=1, na_option='keep')

mom1_rank = mom1_new.copy()
mom1_rank = mom1_rank.rank(axis=1, na_option='keep') # Why出現小數點，有相同股價


# %%  儲存 mom1_rank_reverse

# 將日期倒序排列
mom1_rank_reverse = mom1_rank.T.sort_index(ascending=False)
# 將 DataFrame 寫入 CSV 檔案
mom1_rank_reverse.to_csv(Path_Output+'mom1_rank.csv', index=True, encoding='utf-8')


# %%  計算 IC1 值

corr1_list = []  # 用於儲存所有相關係數的列表

for month in log_returns1_rank.columns:
    log_returns1_monthly = log_returns1_rank[month]                 # 選擇當月的 log_returns_rank 向量
    mom1_monthly = mom1_rank[month]                                 # 選擇當月的 mom1_rank 向量
    corr1 = np.corrcoef(log_returns1_monthly, mom1_monthly)[0, 1]   # 計算相關係數
    corr1_list.append(corr1)

IC1 = pd.DataFrame({'IC1': corr1_list}, index=log_returns1_rank.columns)    # 將相關係數轉換為DataFrame
IC1.to_csv('D:/project2/IC_original/IC1.csv', index=True, encoding='utf-8')


# 將動能資料轉置(Index為時間，column為股票代號，IC已為此格式不須轉置)
mom1_new = mom1_new.T
