import os
import sys
import pandas as pd

from sklearn.metrics import r2_score


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
Path_Input_05 = os.path.join(Path_dir, 'Code/05  輸出資料/')
Path_Output   = os.path.join(Path_dir, 'Code/06  輸出資料/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from filter_by_month_range import filter_by_month_range


# %%  Configuration: Sample Period, Horizon Names and Models

Period_START = '1987-12'
Period_END   = '2024-11'

horizon_name_list = ['01m', '06m', '12m', '36m', '60m']

# All Models:
# ['OLS', 'OLS+H', 'ENet+H', 'PCR', 'PLS', 'GLM+H', 'GBRT+H', 'RF', 'NN']

models = ['OLS', 'ENet+H', 'PCR', 'PLS', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5']


# %%  Import realized IC

dict_IC = {}

for horizon in horizon_name_list:
    fp = os.path.join(Path_Input_02, f'IC_{horizon}.csv')
    df = pd.read_csv(fp, index_col='date')
    
    df.index = pd.to_datetime(df.index).strftime('%Y-%m')
    df.columns = df.columns.astype(str)
    df = df[['Normal_IC']]
    
    dict_IC[f'IC_{horizon}'] = filter_by_month_range(df, Period_START, Period_END)

del fp, df, horizon


# %%  Import predicted IC

def load_pred_table(path_05, horizons, models):
    frames = []
    for h in horizons:
        for m in models:
            fp = os.path.join(path_05, f'predict_Normal_IC_{h}_{m}.csv')
            df = pd.read_csv(fp, usecols=['date', 'pred']).set_index('date')
            df.index = pd.to_datetime(df.index).strftime('%Y-%m')
            df.columns = pd.MultiIndex.from_tuples([(m, h)], names=['model','horizon'])
            frames.append(df)
    wide = pd.concat(frames, axis=1).sort_index()
    return wide

pred_wide = load_pred_table(Path_Input_05, horizon_name_list, models)


# %%  Function of OOS R^2 (without demeaning)

def r2_oos(y_true, y_pred):
    df = pd.concat([y_true, y_pred], axis=1, join='inner').dropna()
    if df.shape[0] == 0:
        return float('nan')
    e2 = ((df.iloc[:,0] - df.iloc[:,1])**2).sum()
    sst = (df.iloc[:,0]**2).sum()
    return 1.0 - e2/sst


# %%

def r2_standard(y_true, y_pred):
    df = pd.concat([y_true, y_pred], axis=1, join='inner').dropna()
    return r2_score(df.iloc[:,0], df.iloc[:,1])


# %%  Compute OOS R^2

R_square = pd.DataFrame(index=horizon_name_list, columns=models, dtype=float)

for h in horizon_name_list:
    y = dict_IC[f'IC_{h}']['Normal_IC']
    for m in models:
        yhat = pred_wide[(m, h)].rename('pred')
        R_square.loc[h, m] = r2_oos(y, yhat) * 100

R_square = R_square.round(2)
R_square.index.name = 'Horizon'

R_square.to_csv(Path_Output+'R_square.csv', index=True)


# %%

R_square_demean = pd.DataFrame(index=horizon_name_list, columns=models, dtype=float)

for h in horizon_name_list:
    y = dict_IC[f'IC_{h}']['Normal_IC']
    for m in models:
        yhat = pred_wide[(m, h)].rename('pred')
        R_square_demean.loc[h, m] = r2_standard(y, yhat) * 100

R_square_demean = R_square_demean.round(2)
R_square_demean.index.name = 'Horizon'


# %%

import matplotlib.pyplot as plt

horizon = '01m'
model = 'RF'
y = dict_IC[f'IC_{horizon}']['Normal_IC']
yhat = pred_wide[(model, horizon)]
df_plot = pd.concat([y, yhat], axis=1, join='inner').dropna()
df_plot.columns = ['True_IC', 'Pred_IC']
plt.scatter(df_plot['True_IC'], df_plot['Pred_IC'], alpha=0.5)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title(f"{model} {horizon}")
plt.show()





