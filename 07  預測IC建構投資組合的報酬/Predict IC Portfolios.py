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
Path_Input_03 = os.path.join(Path_dir, 'Code/03  輸出資料')
Path_Input_05 = os.path.join(Path_dir, 'Code/05  輸出資料')
Path_Output   = os.path.join(Path_dir, 'Code/07  輸出資料')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from filter_by_month_range import filter_by_month_range
from portfolio_strategies  import (
    get_max_abs_IC_factor,
    build_strategies,
    build_cumulative_return
)
from portfolio_analysis    import portfolio_analysis


# %%  Configuration

Period_START = '1987-12'
Period_END   = '2024-11'

horizon_name_list = ['01m', '06m', '12m', '36m', '60m']

percentage_list = [0.01, 0.05, 0.1]
strategy_list   = ['TB_BT']


# %%  Import S&P500

SP500 = pd.read_csv(os.path.join(Path_Input_01, 'SP500.csv'), index_col='date')
SP500 = filter_by_month_range(SP500, Period_START, Period_END)


# %%  Import Top & Bottom monthly Return

dict_Top_Bottom = {}

for horizon in horizon_name_list:
    file_path = os.path.join(Path_Input_03, f"Top_Bottom_monthly_mean_{horizon}.pkl")
    dict_Top_Bottom[horizon] = pd.read_pickle(file_path)

del horizon, file_path


# %%  Import the number of tradable stock

# tradable = pd.read_csv(os.path.join(Path_Input_03, 'monthly_non_nan_counts.csv'), index_col='date')


# %%  Import predicted IC

# All Models:
# ['OLS', 'OLS+H', 'ENet+H', 'PCR', 'PLS', 'GLM+H', 'GBRT+H', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5']
models = ['OLS', 'OLS+H', 'ENet+H', 'PCR', 'PLS', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5']
# models = ['OLS', 'OLS+H', 'ENet+H', 'RF']

for model_name in models:
    print(f'{model_name}')
    
    dict_IC = {}
    for horizon in horizon_name_list:
        fp = os.path.join(Path_Input_05, f'predict_Normal_IC_{horizon}_{model_name}.csv')
        df = pd.read_csv(fp, index_col='date')
        df.index = pd.to_datetime(df.index).strftime('%Y-%m')
        df.columns = df.columns.astype(str)
        dict_IC[f'IC_{horizon}'] = filter_by_month_range(df, Period_START, Period_END)
    
    normal_ic_list = []
    for name, df in dict_IC.items():    
        df_normal = df[['pred']].rename(columns={'pred': name})
        normal_ic_list.append(df_normal)
    
    IC_normal = pd.concat(normal_ic_list, axis=1)
    IC_normal = IC_normal.sort_index(ascending=True).dropna()
    
    max_values_normal = get_max_abs_IC_factor(IC_normal)
    
    # Portfolio performance
    Portfolio_normal = build_strategies(max_values_normal, dict_Top_Bottom, percentage_list, strategy_list)
    
    # Add S&P500 Return
    Portfolio_normal['SP500'] = SP500['SP500_next_period_return']
    Portfolio_T_normal = Portfolio_normal.T
    
    # Analysis
    Portfolio_Analysis_normal = (
        portfolio_analysis(
            Portfolio_T_normal,
            annualize=False,
            strategy_list=strategy_list,
            percentages=percentage_list
        )
    )
    print(Portfolio_Analysis_normal)
    out_path = os.path.join(Path_Output, f'Portfolio_Analysis_Normal_IC_{model_name}.csv')
    Portfolio_Analysis_normal.to_csv(out_path, index=True)
    
    # Calculate cumulative return
    strategy_select = ['TB_BT']
    Portfolio_cumulative_return = build_cumulative_return(Portfolio_T_normal, strategy_select)
    Portfolio_cumulative_return_T = Portfolio_cumulative_return.T
    
    filename = os.path.join(Path_Output, f'Portfolio_Cumulative_Return_Normal_IC_{model_name}.csv')
    Portfolio_cumulative_return_T.to_csv(filename, index=True)
    print(' ')
