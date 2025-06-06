import os
import sys
import pandas as pd


# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)


# %%  Mac 資料夾路徑

Path_Mac = '/Users/irisyu/Library/CloudStorage/GoogleDrive-jouping.yu@gmail.com/'
Path_dir = os.path.join(Path_Mac, Path_PaperFolder)


# %%  Input and Output Path

Path_Input  = os.path.join(Path_dir, 'Code/01  輸出資料/')
Path_Output = os.path.join(Path_dir, 'Code/02  輸出資料/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from Information_Coefficient import Information_Coefficient
from calculate_statistics    import calculate_statistics
from predict_IC              import predict_IC


# %%  Import data

df_stock_price        = pd.read_csv(os.path.join(Path_Input, 'Individual_stock_price.csv'), index_col='permno')
df_next_period_return = pd.read_csv(os.path.join(Path_Input, 'Individual_next_period_return.csv'), index_col='permno')


# %%  計算 momentum、IC 值

mom_01m, mom_01m_rank, IC_01m = Information_Coefficient(df_stock_price, df_next_period_return, 1)
mom_06m, mom_06m_rank, IC_06m = Information_Coefficient(df_stock_price, df_next_period_return, 6)
mom_12m, mom_12m_rank, IC_12m = Information_Coefficient(df_stock_price, df_next_period_return, 12)
mom_36m, mom_36m_rank, IC_36m = Information_Coefficient(df_stock_price, df_next_period_return, 36)
mom_60m, mom_60m_rank, IC_60m = Information_Coefficient(df_stock_price, df_next_period_return, 60)


# %%  Statistics

# Calculate statistics for each DataFrame
IC_01m_stats = calculate_statistics(IC_01m)
IC_06m_stats = calculate_statistics(IC_06m)
IC_12m_stats = calculate_statistics(IC_12m)
IC_36m_stats = calculate_statistics(IC_36m)
IC_60m_stats = calculate_statistics(IC_60m)

# Store in list
all_stats = [IC_01m_stats, IC_06m_stats, IC_12m_stats, IC_36m_stats, IC_60m_stats]
period_names = ['1 month', '6 month', '12 month', '36 month', '60 month']

# 重新組合 columns → MultiIndex columns，展開成「橫向」
all_stats_with_columns = []

for period_name, df_stats in zip(period_names, all_stats):
    df_stats.columns = pd.MultiIndex.from_product([[period_name], df_stats.columns])
    all_stats_with_columns.append(df_stats)

# Concatenate along columns → axis=1
summary_df_horizontal = pd.concat(all_stats_with_columns, axis=1)

# 顯示設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# 顯示
print(summary_df_horizontal)


# %%  匯出表格

mom_01m.to_csv(Path_Output+'mom_01m.csv', index=True, index_label='date')
mom_06m.to_csv(Path_Output+'mom_06m.csv', index=True, index_label='date')
mom_12m.to_csv(Path_Output+'mom_12m.csv', index=True, index_label='date')
mom_36m.to_csv(Path_Output+'mom_36m.csv', index=True, index_label='date')
mom_60m.to_csv(Path_Output+'mom_60m.csv', index=True, index_label='date')

mom_01m_rank.to_csv(Path_Output+'mom_01m_rank.csv', index=True, index_label='date')
mom_06m_rank.to_csv(Path_Output+'mom_06m_rank.csv', index=True, index_label='date')
mom_12m_rank.to_csv(Path_Output+'mom_12m_rank.csv', index=True, index_label='date')
mom_36m_rank.to_csv(Path_Output+'mom_36m_rank.csv', index=True, index_label='date')
mom_60m_rank.to_csv(Path_Output+'mom_60m_rank.csv', index=True, index_label='date')

IC_01m.to_csv(Path_Output+'IC_01m.csv', index=True, index_label='date')
IC_06m.to_csv(Path_Output+'IC_06m.csv', index=True, index_label='date')
IC_12m.to_csv(Path_Output+'IC_12m.csv', index=True, index_label='date')
IC_36m.to_csv(Path_Output+'IC_36m.csv', index=True, index_label='date')
IC_60m.to_csv(Path_Output+'IC_60m.csv', index=True, index_label='date')

summary_df_horizontal.to_csv(Path_Output+'statistics.csv', index=True)


# %%  模型設定

# 設定迴圈次數
n_loops = 100

# 設定初始月份
start_month = 10
start_year  = 2022

# 設定每次迴圈的間隔月份
month_step = -1

# 設定隨機種子以確保結果的可重現性
my_seed = 42


# %%  預測 rank IC

predict_IC_01m = predict_IC(mom_01m, IC_01m, start_year, start_month, month_step, n_loops, my_seed)
predict_IC_06m = predict_IC(mom_06m, IC_06m, start_year, start_month, month_step, n_loops, my_seed)
predict_IC_12m = predict_IC(mom_12m, IC_12m, start_year, start_month, month_step, n_loops, my_seed)
predict_IC_36m = predict_IC(mom_36m, IC_36m, start_year, start_month, month_step, n_loops, my_seed)
predict_IC_60m = predict_IC(mom_60m, IC_60m, start_year, start_month, month_step, n_loops, my_seed)


# %%  匯出表格

predict_IC_01m.to_csv(Path_Output+'predict_IC_01m.csv', index=False)
predict_IC_06m.to_csv(Path_Output+'predict_IC_06m.csv', index=False)
predict_IC_12m.to_csv(Path_Output+'predict_IC_12m.csv', index=False)
predict_IC_36m.to_csv(Path_Output+'predict_IC_36m.csv', index=False)
predict_IC_60m.to_csv(Path_Output+'predict_IC_60m.csv', index=False)
