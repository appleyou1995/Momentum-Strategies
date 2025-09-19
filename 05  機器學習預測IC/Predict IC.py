import os
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
Path_Output   = os.path.join(Path_dir, 'Code/05  輸出資料/')


# %%  Import function

from function_predict_information_coefficient import predict_information_coefficient


# %%  Define horizon_name_list

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

for key, df in dict_IC.items():
    df_Y_Normal[key] = df[['Normal_IC']]

del key, df


# %%  模型設定

# 設定驗證集 window 長度（以月為單位）
valid_length = 120

# 設定測試集開始月份
test_start_date = '1987-12'


# %%  預測 Normal IC

# All Models:
# ['OLS', 'OLS+H', 'ENet+H', 'PCR', 'PLS', 'GLM+H', 'GBRT+H', 'RF', 'NN']

models = ['NN1']

for h in horizon_name_list:
    df_out = predict_information_coefficient(
        dict_mom[f'mom_{h}'],
        df_Y_Normal[f'IC_{h}'],
        test_start_date,
        valid_length,
        models,
        Path_Output,
        h,
        True,
    )


# %%  Sigle horizon R^2

def r2_oos_no_demean(y_true, y_pred):
    sse = ((y_true - y_pred) ** 2).sum()
    sst = (y_true ** 2).sum()
    return 1.0 - sse / sst

df_out = df_out.set_index('date')
trueIC = df_Y_Normal[f'IC_{h}'][['Normal_IC']]
merged = df_out.merge(trueIC, left_index=True, right_index=True, how='left')

y_pred, y_true = merged.iloc[:, 0], merged.iloc[:, 1]

print((r2_oos_no_demean(y_true, y_pred) * 100).round(4))


