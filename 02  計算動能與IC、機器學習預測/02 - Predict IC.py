import os
import sys
import pandas as pd


# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)


# %%  Mac 資料夾路徑

# Path_Mac = '/Users/irisyu/Library/CloudStorage/GoogleDrive-jouping.yu@gmail.com/'
# Path_dir = os.path.join(Path_Mac, Path_PaperFolder)


# %%  Input and Output Path

Path_Input  = os.path.join(Path_dir, 'Code/01  輸出資料/')
Path_Output = os.path.join(Path_dir, 'Code/02  輸出資料/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from predict_IC              import predict_IC


# %%  模型設定

# 設定迴圈次數
n_loops = 100

# 設定每次迴圈的間隔月份
month_step = -1

# 設定隨機種子以確保結果的可重現性
my_seed = 42

# 設定驗證集 window 長度（以月為單位）
window_length = 96

# 設定預測 horizon
horizon = 1


# %%  設定不同 Y 欄位（Normal IC / Rank IC）

df_Y_01m_normal = IC_01m[['Normal_IC']]
df_Y_01m_rank   = IC_01m[['Rank_IC']]

df_Y_06m_normal = IC_06m[['Normal_IC']]
df_Y_06m_rank   = IC_06m[['Rank_IC']]

df_Y_12m_normal = IC_12m[['Normal_IC']]
df_Y_12m_rank   = IC_12m[['Rank_IC']]

df_Y_36m_normal = IC_36m[['Normal_IC']]
df_Y_36m_rank   = IC_36m[['Rank_IC']]

df_Y_60m_normal = IC_60m[['Normal_IC']]
df_Y_60m_rank   = IC_60m[['Rank_IC']]


# %%  預測 IC

# Rank IC
predict_IC_01m_rank = predict_IC(mom_01m, df_Y_01m_rank, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_01m_rank.to_csv(Path_Output + 'predict_IC_01m_rank.csv', index=False)

predict_IC_06m_rank = predict_IC(mom_06m, df_Y_06m_rank, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_06m_rank.to_csv(Path_Output + 'predict_IC_06m_rank.csv', index=False)

predict_IC_12m_rank = predict_IC(mom_12m, df_Y_12m_rank, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_12m_rank.to_csv(Path_Output + 'predict_IC_12m_rank.csv', index=False)

predict_IC_36m_rank = predict_IC(mom_36m, df_Y_36m_rank, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_36m_rank.to_csv(Path_Output + 'predict_IC_36m_rank.csv', index=False)

predict_IC_60m_rank = predict_IC(mom_60m, df_Y_60m_rank, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_60m_rank.to_csv(Path_Output + 'predict_IC_60m_rank.csv', index=False)


# Normal IC
predict_IC_01m_normal = predict_IC(mom_01m, df_Y_01m_normal, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_01m_normal.to_csv(Path_Output + 'predict_IC_01m_normal.csv', index=False)

predict_IC_06m_normal = predict_IC(mom_06m, df_Y_06m_normal, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_06m_normal.to_csv(Path_Output + 'predict_IC_06m_normal.csv', index=False)

predict_IC_12m_normal = predict_IC(mom_12m, df_Y_12m_normal, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_12m_normal.to_csv(Path_Output + 'predict_IC_12m_normal.csv', index=False)

predict_IC_36m_normal = predict_IC(mom_36m, df_Y_36m_normal, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_36m_normal.to_csv(Path_Output + 'predict_IC_36m_normal.csv', index=False)

predict_IC_60m_normal = predict_IC(mom_60m, df_Y_60m_normal, month_step, n_loops, my_seed, window_length, horizon)
predict_IC_60m_normal.to_csv(Path_Output + 'predict_IC_60m_normal.csv', index=False)
