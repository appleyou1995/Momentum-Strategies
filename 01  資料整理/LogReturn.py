import os
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

Path_Input  = os.path.join(Path_dir, 'Data/Raw_Data')
Path_Output = os.path.join(Path_dir, 'Data/')


# %%  讀取資料

df_merged = pd.DataFrame()

# 讀取每個檔案的 1、2 和 47 行資料並合併
for filename in os.listdir(Path_Input):
    #if filename.endswith(".csv"):
    if 'CRSP' in filename:
        # 讀取指定行的資料
        df = pd.read_csv(os.path.join(Path_Input, filename), usecols=[0, 1, 5, 47])
        # 將第三行的數值轉換為浮點數格式
        df.iloc[:, 3] = df.iloc[:, 3].astype(float)
        # 取第三行數值的絕對值
        df.iloc[:, 3] = df.iloc[:, 3].abs()
        # 將讀取到的資料框與已經讀取的資料框合併
        df_merged = pd.concat([df_merged, df], axis=0)


# %%  資料整理與篩選

# 依照日期排序
df_merged = df_merged.sort_values(by=['date', 'PERMNO'])


# 空白補值
df_merged['SICCD'].fillna(method='ffill', inplace=True)


# 刪除 'SICCD' 欄位內非數字值的整列
df_merged = df_merged[pd.to_numeric(df_merged['SICCD'], errors='coerce').notnull()]
df_merged.iloc[:, 2] = df_merged.iloc[:, 2].astype(int)


# 刪除 SICCD 值在 4900~4999 和 6000~6999 之間的整列
df_merged = df_merged[(df_merged['SICCD'] <= 4900) | (df_merged['SICCD'] >= 4999)]
df_merged = df_merged[(df_merged['SICCD'] <= 6000) | (df_merged['SICCD'] >= 6999)]


# %%  將 df_merged 轉換為以 PERMNO 為索引，日期為列名的新資料框

df_new = pd.pivot_table(df_merged, index='PERMNO', columns='date', values='PRC')


# 將 df_new 的 column 改成 DatetimeIndex
df_new.columns = pd.to_datetime(df_new.columns).strftime("%Y-%m")


# 修正 2022-10、2022-11、2022-12 錯位
col_names = sorted(list(df_new.columns))
df_new = df_new[col_names[:]]


########################### 待確認 ###########################
# 2022/12 已下市的股票刪除
df_new = df_new.dropna(subset=df_new.columns[-1], how='any')


# 將所有股價轉為浮點數
df_new = df_new.apply(pd.to_numeric, errors='coerce')


# %%  計算每間公司每個月的 logreturn (ln(t+1/t))

log_returns = np.log(df_new.shift(-1, axis=1) / df_new)


# 將日期倒序排列
log_returns_reverse = log_returns.T
log_returns_reverse = log_returns_reverse.sort_index(ascending=False)


# %%  # 將 DataFrame 寫入 CSV 檔案

df_new.to_csv(Path_Output+'Stock_price.csv', index=True, encoding='utf-8')
log_returns.to_csv(Path_Output+'log_returns.csv', index=True, encoding='utf-8')
log_returns_reverse.to_csv(Path_Output+'log_returns_reverse.csv', index=True, encoding='utf-8')
