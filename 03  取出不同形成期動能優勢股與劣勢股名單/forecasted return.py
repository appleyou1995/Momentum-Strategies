import os
import sys
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

Path_Input_01  = os.path.join(Path_dir, 'Code/01  輸出資料/')
Path_Input_02  = os.path.join(Path_dir, 'Code/02  輸出資料/')
Path_Output = os.path.join(Path_dir, 'Code/03  輸出資料/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from Top_Bottom_Return import Top_Bottom_Return


# %%  Import data

stock_price = pd.read_csv(os.path.join(Path_Input_01, 'Individual_stock_price.csv'), index_col='permno')
stock_price = stock_price.T
stock_price.index = pd.to_datetime(stock_price.index).strftime('%Y-%m')

log_return = pd.read_csv(os.path.join(Path_Input_01, 'Individual_next_period_return.csv'), index_col='permno')
log_return = log_return.T
log_return.index = pd.to_datetime(log_return.index).strftime('%Y-%m')


# %%  Import Momentum Rank

mom_01m_rank = pd.read_csv(os.path.join(Path_Input_02, 'mom_01m_rank.csv'), index_col='date')
mom_06m_rank = pd.read_csv(os.path.join(Path_Input_02, 'mom_06m_rank.csv'), index_col='date')
mom_12m_rank = pd.read_csv(os.path.join(Path_Input_02, 'mom_12m_rank.csv'), index_col='date')
mom_36m_rank = pd.read_csv(os.path.join(Path_Input_02, 'mom_36m_rank.csv'), index_col='date')
mom_60m_rank = pd.read_csv(os.path.join(Path_Input_02, 'mom_60m_rank.csv'), index_col='date')


# %%  Top_Bottom_Return

# 設定 pct 組合
pct_list = [0.01, 0.02, 0.1]

# 對應的 mom rank 和檔名 prefix
mom_rank_list = [mom_01m_rank, mom_06m_rank, mom_12m_rank, mom_36m_rank, mom_60m_rank]
horizon_name_list = ['01m', '06m', '12m', '36m', '60m']


# %% 迴圈跑不同 pct

for pct in pct_list:
    
    print(f"Running Top_Bottom_Return with pct = {pct:.0%}")
    
    for mom_rank, horizon_name in zip(mom_rank_list, horizon_name_list):
        
        # Run Top_Bottom_Return
        result, Top_average, Bottom_average, Top_stocks, Bottom_stocks = Top_Bottom_Return(
            mom_rank, stock_price, log_return, pct=pct)
        
        # 檔名用 pct 百分比轉成 1, 2, 10 儲存
        pct_name = str(int(pct * 100))
        
        # 匯出 csv
        result.to_csv(Path_Output + f'result_{horizon_name}_{pct_name}pct.csv', index=True, index_label='date')
        Top_average.to_csv(Path_Output + f'Top_average_{horizon_name}_{pct_name}pct.csv', index=True, index_label='date')
        Bottom_average.to_csv(Path_Output + f'Bottom_average_{horizon_name}_{pct_name}pct.csv', index=True, index_label='date')
        Top_stocks.to_csv(Path_Output + f'Top_stocks_{horizon_name}_{pct_name}pct.csv', index=True, index_label='date')
        Bottom_stocks.to_csv(Path_Output + f'Bottom_stocks_{horizon_name}_{pct_name}pct.csv', index=True, index_label='date')
