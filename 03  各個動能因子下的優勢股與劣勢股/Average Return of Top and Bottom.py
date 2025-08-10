import os
import sys
import pandas as pd
import numpy  as np


# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)


# %%  Mac 資料夾路徑

# Path_Mac = '/Users/irisyu/Library/CloudStorage/GoogleDrive-jouping.yu@gmail.com/'
# Path_dir = os.path.join(Path_Mac, Path_PaperFolder)


# %%  Input and Output Path

Path_Input_01  = os.path.join(Path_dir, 'Code/01  輸出資料/')
Path_Input_02  = os.path.join(Path_dir, 'Code/02  輸出資料/')
Path_Output    = os.path.join(Path_dir, 'Code/03  輸出資料/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from Top_Bottom_monthly_means import Top_Bottom_monthly_means
from keep_month_range         import keep_month_range


# %%  Import data

# 讀取資料並轉置（使列為時間、行為PERMNO）
stock_price = pd.read_csv(os.path.join(Path_Input_01, 'Individual_stock_price_manual.csv'), index_col='PERMNO').T

# 將 index 轉為 datetime 格式並格式化為年月字串；將 column 轉為字串
stock_price.index = pd.to_datetime(stock_price.index).strftime('%Y-%m')
stock_price.columns = stock_price.columns.astype(str)

# 計算 log return: ln(P_{t+1} / P_t)
log_return = np.log(stock_price.shift(-1) / stock_price)


# %%  Import Momentum

mom_01m = pd.read_csv(os.path.join(Path_Input_02, 'mom_01m.csv'), index_col='date')
mom_06m = pd.read_csv(os.path.join(Path_Input_02, 'mom_06m.csv'), index_col='date')
mom_12m = pd.read_csv(os.path.join(Path_Input_02, 'mom_12m.csv'), index_col='date')
mom_36m = pd.read_csv(os.path.join(Path_Input_02, 'mom_36m.csv'), index_col='date')
mom_60m = pd.read_csv(os.path.join(Path_Input_02, 'mom_60m.csv'), index_col='date')

mom_01m.index = pd.to_datetime(mom_01m.index).strftime('%Y-%m')
mom_06m.index = pd.to_datetime(mom_06m.index).strftime('%Y-%m')
mom_12m.index = pd.to_datetime(mom_12m.index).strftime('%Y-%m')
mom_36m.index = pd.to_datetime(mom_36m.index).strftime('%Y-%m')
mom_60m.index = pd.to_datetime(mom_60m.index).strftime('%Y-%m')


# %%  將時間切齊

Period_START = '1987-12'
Period_END   = '2024-11'

(stock_price, log_return, mom_01m, mom_06m, mom_12m, mom_36m, mom_60m) = [
    keep_month_range(df, Period_START, Period_END) for df in
    [stock_price, log_return, mom_01m, mom_06m, mom_12m, mom_36m, mom_60m]
]


# %%  檢查

dfs = [stock_price, log_return, mom_01m, mom_06m, mom_12m, mom_36m, mom_60m]
names = ['stock_price', 'log_return', 'mom_01m', 'mom_06m', 'mom_12m', 'mom_36m', 'mom_60m']

# 排序欄位
dfs = [df[sorted(df.columns)] for df in dfs]
stock_price, log_return, mom_01m, mom_06m, mom_12m, mom_36m, mom_60m = dfs

# 先確保 index 和 columns 完全一致
for df in dfs[1:]:
    assert stock_price.index.equals(df.index), "Index 不一致"
    assert stock_price.columns.equals(df.columns), "Columns 不一致"

count_series = [df.notna().sum(axis=1) for df in dfs]
counts_df = pd.concat(count_series, axis=1)
counts_df.columns = names


# %%  Filter return

log_return_tradable = log_return.where(mom_01m.notna())


# %%  Calculate the number of tradable stock

monthly_non_nan_counts = mom_01m.count(axis=1).to_frame(name='non_nan_count')

csv_path = os.path.join(Path_Output, 'monthly_non_nan_counts.csv')
monthly_non_nan_counts.to_csv(csv_path)


# %%  Plot the number of tradable stock

import matplotlib.pyplot as plt

monthly_non_nan_counts = mom_01m.count(axis=1)

ax = monthly_non_nan_counts.plot(figsize=(7, 5), 
                                 title='Number of Non-NaN Momentum per Month')
ax.set_xlabel("Month")
ax.set_ylabel("Number of Stocks")

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

fig_path = os.path.join(Path_Output, 'monthly_non_nan_counts.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')

plt.show()


# %%  Top & Bottom monthly Return

# 設定 percentage 組合
percentage_list = [0.01, 0.05, 0.1]

# 對應的 mom rank 和檔名 prefix
mom_list = [mom_01m, mom_06m, mom_12m, mom_36m, mom_60m]
horizon_name_list = ['01m', '06m', '12m', '36m', '60m']

os.makedirs(Path_Output, exist_ok=True)


# %%  Output result to pickle

for mom_df, horizon in zip(mom_list, horizon_name_list):
    result = Top_Bottom_monthly_means(log_return_tradable, mom_df, percentage_list)
    file_path = os.path.join(Path_Output, f"Top_Bottom_monthly_mean_{horizon}.pkl")
    result.to_pickle(file_path)

