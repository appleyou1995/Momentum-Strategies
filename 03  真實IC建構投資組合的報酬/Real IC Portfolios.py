import os
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

Path_Input  = os.path.join(Path_dir, 'Code/02  輸出資料')
Path_Output = os.path.join(Path_dir, 'Code/03  輸出資料')


# %%  Import real IC

IC_01m = pd.read_csv(os.path.join(Path_Input, 'IC_01m.csv'), index_col='date')
IC_06m = pd.read_csv(os.path.join(Path_Input, 'IC_06m.csv'), index_col='date')
IC_12m = pd.read_csv(os.path.join(Path_Input, 'IC_12m.csv'), index_col='date')
IC_36m = pd.read_csv(os.path.join(Path_Input, 'IC_36m.csv'), index_col='date')
IC_60m = pd.read_csv(os.path.join(Path_Input, 'IC_60m.csv'), index_col='date')


# %%  合併所有 IC

# 先把原始 5 個 IC DataFrame 放進 dictionary
IC_dict = {
    'IC_01m': IC_01m,
    'IC_06m': IC_06m,
    'IC_12m': IC_12m,
    'IC_36m': IC_36m,
    'IC_60m': IC_60m
}

# 建兩個空 list，等下用來收要 concat 的 DataFrame
normal_ic_list = []
rank_ic_list = []

for name, df in IC_dict.items():
    
    # Normal IC
    df_normal = df[['Normal_IC']].rename(columns={'Normal_IC': name})
    normal_ic_list.append(df_normal)
    
    # Rank IC
    df_rank = df[['Rank_IC']].rename(columns={'Rank_IC': name})
    rank_ic_list.append(df_rank)
    
del df_normal, df_rank, name, df


# 合併成大 DataFrame
IC_normal = pd.concat(normal_ic_list, axis=1)
IC_rank   = pd.concat(rank_ic_list, axis=1)

# 排序 & dropna
IC_normal = IC_normal.sort_index(ascending=False).dropna()
IC_rank   = IC_rank.sort_index(ascending=False).dropna()

# index 轉成 YYYY-MM
IC_normal.index = pd.to_datetime(IC_normal.index).strftime('%Y-%m')
IC_rank.index   = pd.to_datetime(IC_rank.index).strftime('%Y-%m')


# %%  計算 Normal IC 的 argmax

max_columns_normal = IC_normal.abs().idxmax(axis=1)

max_values_normal = pd.DataFrame({
    "max_column": max_columns_normal,
    "max_value": IC_normal.abs().max(axis=1),
})

# 取原始值
max_values_normal["original_value"] = max_values_normal.apply(
    lambda row: IC_normal.loc[row.name, row["max_column"]], axis=1
)

# 排序
max_values_normal = max_values_normal.sort_index(ascending=False)


# %%  計算 Rank IC 的 argmax

max_columns_rank = IC_rank.abs().idxmax(axis=1)

max_values_rank = pd.DataFrame({
    "max_column": max_columns_rank,
    "max_value": IC_rank.abs().max(axis=1),
})

# 取原始值
max_values_rank["original_value"] = max_values_rank.apply(
    lambda row: IC_rank.loc[row.name, row["max_column"]], axis=1
)

# 排序
max_values_rank = max_values_rank.sort_index(ascending=False)


# %%  Import Momentum

mom_01m = pd.read_csv(os.path.join(Path_Input, 'mom_01m.csv'), index_col='date')
mom_06m = pd.read_csv(os.path.join(Path_Input, 'mom_06m.csv'), index_col='date')
mom_12m = pd.read_csv(os.path.join(Path_Input, 'mom_12m.csv'), index_col='date')
mom_36m = pd.read_csv(os.path.join(Path_Input, 'mom_36m.csv'), index_col='date')
mom_60m = pd.read_csv(os.path.join(Path_Input, 'mom_60m.csv'), index_col='date')

df_dict = {'mom_01m': mom_01m, 
           'mom_06m': mom_06m, 
           'mom_12m': mom_12m, 
           'mom_36m': mom_36m, 
           'mom_60m': mom_60m}




