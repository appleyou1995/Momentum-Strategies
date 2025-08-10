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

Path_Input_01 = os.path.join(Path_dir, 'Code/01  輸出資料')
Path_Input_02 = os.path.join(Path_dir, 'Code/02  輸出資料')
Path_Input_03 = os.path.join(Path_dir, 'Code/03  輸出資料')
Path_Output   = os.path.join(Path_dir, 'Code/04  輸出資料')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from keep_month_range import keep_month_range
from build_portfolio  import build_portfolio


# %%

Period_START = '1987-12'
Period_END   = '2024-11'

horizon_name_list = ['01m', '06m', '12m', '36m', '60m']


# %%  Import S&P500

SP500 = pd.read_csv(os.path.join(Path_Input_01, 'SP500.csv'), index_col='date')
SP500 = keep_month_range(SP500, Period_START, Period_END)


# %%  Import real IC

dict_IC = {}

for horizon in horizon_name_list:
    fp = os.path.join(Path_Input_02, f'IC_{horizon}.csv')
    df = pd.read_csv(fp, index_col='date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m')
    df.columns = df.columns.astype(str)
    dict_IC[f'IC_{horizon}'] = keep_month_range(df, Period_START, Period_END)

del fp, df, horizon

# 建兩個空 list，等下用來收要 concat 的 DataFrame
normal_ic_list = []
rank_ic_list = []

for name, df in dict_IC.items():
    
    # Normal IC
    df_normal = df[['Normal_IC']].rename(columns={'Normal_IC': name})
    normal_ic_list.append(df_normal)
    
    # Rank IC
    df_rank = df[['Rank_IC']].rename(columns={'Rank_IC': name})
    rank_ic_list.append(df_rank)
    
del df, df_normal, df_rank, name

# 合併成大 DataFrame
IC_normal = pd.concat(normal_ic_list, axis=1)
IC_rank   = pd.concat(rank_ic_list, axis=1)

# 排序 & dropna
IC_normal = IC_normal.sort_index(ascending=True).dropna()
IC_rank   = IC_rank.sort_index(ascending=True).dropna()

del normal_ic_list, rank_ic_list


# %%  函數：Normal & Rank IC 每月取絕對值後的最大值

def get_max_ic_info(IC_df):    
    max_columns = IC_df.abs().idxmax(axis=1)
    max_values = pd.DataFrame({
        "max_abs_column": max_columns,
        "max_abs_value": IC_df.abs().max(axis=1),
    })
    max_values["original_value"] = max_values.apply(
        lambda row: IC_df.loc[row.name, row["max_abs_column"]], axis=1
    )
    return max_values.sort_index(ascending=True)


# %%  計算 Normal & Rank IC 每月取絕對值後的最大值

max_values_normal = get_max_ic_info(IC_normal)
max_values_rank   = get_max_ic_info(IC_rank)


# %%  Import Momentum

# dict_mom = {}

# for horizon in horizon_name_list:
#     fp = os.path.join(Path_Input_02, f'mom_{horizon}.csv')
#     df = pd.read_csv(fp, index_col='date')
    
#     df.index = pd.to_datetime(df.index).strftime('%Y-%m')
#     df.columns = df.columns.astype(str)
    
#     dict_mom[f'mom_{horizon}'] = keep_month_range(df, Period_START, Period_END)


# %%  Import Top & Bottom monthly Return

dict_Top_Bottom = {}

for horizon in horizon_name_list:
    file_path = os.path.join(Path_Input_03, f"Top_Bottom_monthly_mean_{horizon}.pkl")
    dict_Top_Bottom[horizon] = pd.read_pickle(file_path)

del horizon, file_path


# %%  Import the number of tradable stock

tradable = pd.read_csv(os.path.join(Path_Input_03, 'monthly_non_nan_counts.csv'), index_col='date')

max_values_normal['tradable'] = tradable['non_nan_count']
max_values_rank['tradable']   = tradable['non_nan_count']


# %%  Portfolio performance

percentage_list = [0.01, 0.05, 0.1]

Portfolio_normal = build_portfolio(max_values_normal, dict_Top_Bottom, percentage_list)
Portfolio_rank   = build_portfolio(max_values_rank,   dict_Top_Bottom, percentage_list)


# %%  Analysis

Portfolio_normal['SP500'] = SP500['SP500_next_period_return']
Portfolio_rank['SP500']   = SP500['SP500_next_period_return']

Portfolio_T_normal = Portfolio_normal.T
Portfolio_T_rank   = Portfolio_rank.T


# %%  Calculate sharpe ratio

def portfolio_analysis(df_T, annualize, periods_per_year=12):

    rows = ['tradable', 'portfolio_1%', 'portfolio_5%', 'portfolio_10%', 'SP500']

    # 只取數值欄位（各月份）
    vals = df_T.loc[rows].apply(pd.to_numeric, errors='coerce')

    means = vals.mean(axis=1)
    stds  = vals.std(axis=1, ddof=1)

    # Sharpe（對 returns 列計算；tradable 設為 NaN）
    sharpe = means / stds
    sharpe.loc['tradable'] = np.nan

    if annualize:
        # 年化 mean/std 的標準作法：Sharpe_annual = Sharpe * sqrt(periods_per_year)
        sharpe = sharpe * np.sqrt(periods_per_year)

    out = pd.DataFrame({
        'mean': means,
        'std': stds,
        'sharpe': sharpe
    }).loc[rows]

    return out


Portfolio_Analysis_normal = portfolio_analysis(Portfolio_T_normal, annualize=False).astype(float).round(4)
Portfolio_Analysis_rank   = portfolio_analysis(Portfolio_T_rank,   annualize=False).astype(float).round(4)


# %%  Whether the portfolio outperforms the S&P 500

def portfolio_vs_sp500(df_T):

    portfolios = ['portfolio_1%', 'portfolio_5%', 'portfolio_10%']
    
    # 取 SP500 的數值列
    sp500_row = df_T.loc['SP500']
    
    # 比較，大於為 True(轉成1)，否則 False(轉成0)
    comparison_df = df_T.loc[portfolios].gt(sp500_row, axis=1).astype(int)
    comparison_df['outperform_ratio'] = comparison_df.mean(axis=1)
    
    return comparison_df


Outperform_normal = portfolio_vs_sp500(Portfolio_T_normal)
Outperform_rank   = portfolio_vs_sp500(Portfolio_T_rank)


# %%  

Portfolio_Analysis_normal = Portfolio_Analysis_normal.join(Outperform_normal['outperform_ratio']).astype(float).round(4)
Portfolio_Analysis_rank = Portfolio_Analysis_rank.join(Outperform_rank['outperform_ratio']).astype(float).round(4)

Portfolio_Analysis_normal.to_csv(os.path.join(Path_Output, 'Portfolio_Analysis_normal.csv'))
Portfolio_Analysis_rank.to_csv(os.path.join(Path_Output, 'Portfolio_Analysis_rank.csv'))











