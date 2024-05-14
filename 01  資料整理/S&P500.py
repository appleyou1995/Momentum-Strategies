# S&P500資料整理

import os
import pandas as pd
import numpy as np


# %%  論文資料夾路徑

Paper_folder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Win      = ''
file_dir = os.path.join(Win, Paper_folder)


# %%  Mac 資料夾路徑

Mac      = '/Users/irisyu/Library/CloudStorage/GoogleDrive-jouping.yu@gmail.com/'
file_dir = os.path.join(Mac, Paper_folder)


# %%  Input and Output Path

Input_dir  = os.path.join(file_dir, 'Data/Raw_Data')
Output_dir = os.path.join(file_dir, 'Data/')


# %%  讀取資料

# 初始化資料框
df_merged = pd.DataFrame()

# 讀取檔案的 1 和 4 行資料
for filename in os.listdir(Input_dir):
    if "S&P500.csv" in filename:
        df = pd.read_csv(os.path.join(Input_dir, filename), usecols=[0, 4])
        # 將包含千位分隔符的字串轉換為浮點數
        df["收盤價"] = df["收盤價"].str.replace(",", "").astype(float)
        df.iloc[:, 1] = df.iloc[:, 1].astype(float)
        df_merged = pd.concat([df_merged, df], axis=0)


# %%  資料格式調整

# 將日期欄位轉換為日期型別
df_merged['日期'] = pd.to_datetime(df_merged['日期']) 
df_new = pd.pivot_table(df_merged, index='日期')

# 將df_new的Index改成DatetimeIndex
df_new.index = pd.to_datetime(df_new.index).strftime("%Y-%m")

# 將所有收盤價轉為浮點數
df_new = df_new.apply(pd.to_numeric, errors='coerce')


# %%  生成 SP500_log_returns 檔案

# 計算log return(從1995-01開始有值)(ln(t+1/t))
SP500_log_returns = np.log(df_new['收盤價'].shift(-1)/ df_new['收盤價'])

# 將日期倒序排列
SP500_log_returns = SP500_log_returns.sort_index(ascending=False)

# 將 "收盤價" 改名為 "SP500_logreturn"。
SP500_log_returns = SP500_log_returns.rename('SP500_logreturn')
SP500_log_returns = SP500_log_returns.rename_axis('date')

# 將 DataFrame 寫入 CSV 檔案
SP500_log_returns.to_csv(Output_dir+'/SP500_logreturns.csv', index=True, encoding='utf-8')


# %% SP500報酬率處理

# 讀取七個檔案並設定 index
SP500 = pd.read_csv(Output_dir+'SP500_logreturns.csv').set_index('date')
SP500.index = pd.to_datetime(SP500.index).strftime("%Y-%m")
SP500 = SP500.rename(columns={'SP500_logreturn': 'SP500'})

# 計算累積報酬率
reversed_SP500 = SP500.iloc[::-1]
cumulative_SP500 = pd.DataFrame()
cumulative_SP500['SP500'] = []
cumulative_SP500['SP500'] = reversed_SP500['SP500'].cumsum()

# 年化報酬率
cumulative_SP5001 = cumulative_SP500.copy()
cumulative_SP5001['i'] = range(1, 101)
cumulative_SP5001['Linear'] = cumulative_SP5001['SP500'].astype(float)
cumulative_SP5001['i'] = cumulative_SP5001['i'].astype(float)

def calculate_formula(x, i):
    return (x / (i / 12)) 

IRR_SP500 = pd.DataFrame()
IRR_SP500['SP500'] = []
IRR_SP500['SP500'] = cumulative_SP5001.apply(lambda row: calculate_formula(row['SP500'], row['i']), axis=1)

# 計算累積報酬率
reversed_SP500 = SP500.iloc[::-1]
cumulative_SP500 = (1 + reversed_SP500).cumprod() - 1


IRR_SP500.to_csv(Output_dir+'IRR_SP500.csv', index=True, encoding='utf-8')
cumulative_SP500.to_csv(Output_dir+'cumulative_SP500.csv', index=True, encoding='utf-8')
