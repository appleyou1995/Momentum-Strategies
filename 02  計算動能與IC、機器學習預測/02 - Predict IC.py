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

Path_Input_02 = os.path.join(Path_dir, 'Code/02  輸出資料/')
Path_Output   = os.path.join(Path_dir, 'Code/02  輸出資料/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from predict_information_coefficient import predict_information_coefficient


# %%

# Period_START = '1987-12'
# Period_END   = '2024-11'

horizon_name_list = ['01m', '06m', '12m', '36m', '60m']


# %%  Import Momentum

dict_mom = {}

for horizon in horizon_name_list:
    fp = os.path.join(Path_Input_02, f'mom_{horizon}.csv')
    df = pd.read_csv(fp, index_col='date')
    
    df.index = pd.to_datetime(df.index).strftime('%Y-%m')
    df.columns = df.columns.astype(str)
    
    # dict_mom[f'mom_{horizon}'] = filter_by_month_range(df, Period_START, Period_END)
    dict_mom[f'mom_{horizon}'] = df

del fp, df, horizon


# %%  Import IC

dict_IC = {}

for horizon in horizon_name_list:
    fp = os.path.join(Path_Input_02, f'IC_{horizon}.csv')
    df = pd.read_csv(fp, index_col='date')
    
    df.index = pd.to_datetime(df.index).strftime('%Y-%m')
    df.columns = df.columns.astype(str)
    
    # dict_IC[f'IC_{horizon}'] = filter_by_month_range(df, Period_START, Period_END)
    dict_IC[f'IC_{horizon}'] = df

del fp, df, horizon


# %%  設定不同 Y 欄位（Normal IC / Rank IC）

df_Y_Normal = {}
df_Y_Rank   = {}

for key, df in dict_IC.items():
    df_Y_Normal[key] = df[['Normal_IC']]
    df_Y_Rank[key]   = df[['Rank_IC']]

del key, df


# %%  模型設定

# 設定每次迴圈的間隔月份
month_step = -1

# 設定迴圈次數
n_loops = 100

# 設定隨機種子以確保結果的可重現性
seed = 999

# 設定驗證集 window 長度（以月為單位）
window_length = 120

# 設定預測 horizon
horizon = 1


# %%  預測 IC

# Normal IC
for h in horizon_name_list:
    df_pred = predict_information_coefficient(
        dict_mom[f'mom_{h}'],
        df_Y_Normal[f'IC_{h}'],
        month_step,
        n_loops,
        seed,
        window_length,
        horizon
    )

    df_pred.to_csv(Path_Output + f'predict_Normal_IC_{h}.csv', index=False)


# Rank IC
for h in horizon_name_list:
    df_pred = predict_information_coefficient(
        dict_mom[f'mom_{h}'],
        df_Y_Rank[f'IC_{h}'],
        month_step,
        n_loops,
        seed,
        window_length,
        horizon
    )

    df_pred.to_csv(Path_Output + f'predict_Rank_IC_{h}.csv', index=False)


