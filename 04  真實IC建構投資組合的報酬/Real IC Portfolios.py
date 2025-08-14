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

Path_Input_01 = os.path.join(Path_dir, 'Code/01  輸出資料')
Path_Input_02 = os.path.join(Path_dir, 'Code/02  輸出資料')
Path_Input_03 = os.path.join(Path_dir, 'Code/03  輸出資料')
Path_Output   = os.path.join(Path_dir, 'Code/04  輸出資料')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from filter_by_month_range import filter_by_month_range
from portfolio_strategies  import (
    get_max_abs_IC_factor,
    build_strategies,
    build_cumulative_return,
    build_reinvested_return
)
from portfolio_analysis    import portfolio_analysis
from plot                  import (
    configure, 
    plot_cumulative_return, 
    plot_reinvested_return
)


# %%

Period_START = '1987-12'
Period_END   = '2024-11'

horizon_name_list = ['01m', '06m', '12m', '36m', '60m']


# %%  Import S&P500

SP500 = pd.read_csv(os.path.join(Path_Input_01, 'SP500.csv'), index_col='date')
SP500 = filter_by_month_range(SP500, Period_START, Period_END)


# %%  Import real IC

dict_IC = {}

for horizon in horizon_name_list:
    fp = os.path.join(Path_Input_02, f'IC_{horizon}.csv')
    df = pd.read_csv(fp, index_col='date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m')
    df.columns = df.columns.astype(str)
    dict_IC[f'IC_{horizon}'] = filter_by_month_range(df, Period_START, Period_END)

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


# %%  計算 Normal & Rank IC 每月取絕對值後的最大值

max_values_normal = get_max_abs_IC_factor(IC_normal)
max_values_rank   = get_max_abs_IC_factor(IC_rank)


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

strategy_list = [
    'TB_BT',         # 1. TB / BT
    'BuyT_BuyB',     # 2. BuyT / BuyB
    'BuyT_SellB',    # 3. BuyT / SellB
    'SellB_SellT',   # 4. SellB / SellT
    'BuyT',          # 5. BuyT
    'BuyB',          # 6. BuyB
    'SellT',         # 7. SellT
    'SellB',         # 8. SellB
    'BuyTSellB'      # 9. BuyT and SellB simultaneously
]

Portfolio_normal = build_strategies(max_values_normal, dict_Top_Bottom, percentage_list, strategy_list)
Portfolio_rank   = build_strategies(max_values_rank,   dict_Top_Bottom, percentage_list, strategy_list)


# %%  Add S&P500 Return

Portfolio_normal['SP500'] = SP500['SP500_next_period_return']
Portfolio_rank['SP500']   = SP500['SP500_next_period_return']

Portfolio_T_normal = Portfolio_normal.T
Portfolio_T_rank   = Portfolio_rank.T


# %%  Analysis

Portfolio_Analysis_normal = (
    portfolio_analysis(
        Portfolio_T_normal,
        annualize=False,
        strategy_list=strategy_list,
        percentages=percentage_list
    )
)

Portfolio_Analysis_rank = (
    portfolio_analysis(
        Portfolio_T_rank,
        annualize=False,
        strategy_list=strategy_list,
        percentages=percentage_list
    ).astype(float).round(4)
)


# %%  Output: dataframe

Portfolio_Analysis_normal.to_csv(os.path.join(Path_Output, 'Portfolio_Analysis_normal.csv'))
Portfolio_Analysis_rank.to_csv(os.path.join(Path_Output, 'Portfolio_Analysis_rank.csv'))


# %%  Calculate cumulative return and reinvested return

strategy_plot = ['TB_BT']
Portfolio_cumulative_return = build_cumulative_return(Portfolio_T_normal, strategy_plot)
Portfolio_reinvested_return = build_reinvested_return(Portfolio_T_normal, strategy_plot)


# %%  Plot cumulative return

configure()
fig_cumulative, ax = plot_cumulative_return(Portfolio_cumulative_return)


# %%  Plot reinvested return

configure()
fig_reinvested, ax = plot_reinvested_return(Portfolio_reinvested_return)


# %%  Output: plot

strategy_str = "_".join(strategy_plot)

filename = os.path.join(Path_Output, f"Plot_Real_IC_cumulative_return_{strategy_str}.pdf")
fig_cumulative.savefig(filename, format="pdf")

filename = os.path.join(Path_Output, f"Plot_Real_IC_reinvested_return_{strategy_str}.pdf")
fig_reinvested.savefig(filename, format="pdf")
