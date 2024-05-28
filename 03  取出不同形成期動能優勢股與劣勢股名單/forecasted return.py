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

Path_Input  = os.path.join(Path_dir, 'Data/')
Path_Output = os.path.join(Path_dir, 'Data/Portfolio/')


# %%  Import function

sys.path.append(Path_dir+'/Code/99  自訂函數')

from Top_Bottom_Return import Top_Bottom_Return


# %%  Import data

stock_price = pd.read_csv(os.path.join(Path_Input, 'Stock_price.csv'), index_col='PERMNO')
stock_price = stock_price.T
stock_price.index = pd.to_datetime(stock_price.index).strftime('%Y-%m')

log_return  = pd.read_csv(os.path.join(Path_Input, 'log_returns.csv'), index_col='PERMNO')
log_return = log_return.T
log_return.index = pd.to_datetime(log_return.index).strftime('%Y-%m')


# %%  Import Momentum Rank

mom_01m_rank = pd.read_csv(os.path.join(Path_Input, 'mom_01m_rank.csv'), index_col='date')
mom_06m_rank = pd.read_csv(os.path.join(Path_Input, 'mom_06m_rank.csv'), index_col='date')
mom_12m_rank = pd.read_csv(os.path.join(Path_Input, 'mom_12m_rank.csv'), index_col='date')
mom_36m_rank = pd.read_csv(os.path.join(Path_Input, 'mom_36m_rank.csv'), index_col='date')
mom_60m_rank = pd.read_csv(os.path.join(Path_Input, 'mom_60m_rank.csv'), index_col='date')


# %%  Top_Bottom_Return

# num=1：只挑一支股票

(result_01m, 
 Top_average_01m, 
 Bottom_average_01m, 
 Top_stocks_01m, 
 Bottom_stocks_01m) = Top_Bottom_Return(mom_01m_rank, stock_price, log_return, num=1)

(result_06m, 
 Top_average_06m, 
 Bottom_average_06m, 
 Top_stocks_06m, 
 Bottom_stocks_06m) = Top_Bottom_Return(mom_06m_rank, stock_price, log_return, num=1)

(result_12m, 
 Top_average_12m, 
 Bottom_average_12m, 
 Top_stocks_12m, 
 Bottom_stocks_12m) = Top_Bottom_Return(mom_12m_rank, stock_price, log_return, num=1)

(result_36m, 
 Top_average_36m, 
 Bottom_average_36m, 
 Top_stocks_36m, 
 Bottom_stocks_36m) = Top_Bottom_Return(mom_36m_rank, stock_price, log_return, num=1)

(result_60m, 
 Top_average_60m, 
 Bottom_average_60m, 
 Top_stocks_60m, 
 Bottom_stocks_60m) = Top_Bottom_Return(mom_60m_rank, stock_price, log_return, num=1)


# %%  Ouput dataframe

result_01m.to_csv(Path_Output+'result_01m.csv', index=True, index_label='date')
result_06m.to_csv(Path_Output+'result_06m.csv', index=True, index_label='date')
result_12m.to_csv(Path_Output+'result_12m.csv', index=True, index_label='date')
result_36m.to_csv(Path_Output+'result_36m.csv', index=True, index_label='date')
result_60m.to_csv(Path_Output+'result_60m.csv', index=True, index_label='date')

Top_average_01m.to_csv(Path_Output+'Top_average_01m.csv', index=True, index_label='date')
Top_average_06m.to_csv(Path_Output+'Top_average_06m.csv', index=True, index_label='date')
Top_average_12m.to_csv(Path_Output+'Top_average_12m.csv', index=True, index_label='date')
Top_average_36m.to_csv(Path_Output+'Top_average_36m.csv', index=True, index_label='date')
Top_average_60m.to_csv(Path_Output+'Top_average_60m.csv', index=True, index_label='date')

Bottom_average_01m.to_csv(Path_Output+'Bottom_average_01m.csv', index=True, index_label='date')
Bottom_average_06m.to_csv(Path_Output+'Bottom_average_06m.csv', index=True, index_label='date')
Bottom_average_12m.to_csv(Path_Output+'Bottom_average_12m.csv', index=True, index_label='date')
Bottom_average_36m.to_csv(Path_Output+'Bottom_average_36m.csv', index=True, index_label='date')
Bottom_average_60m.to_csv(Path_Output+'Bottom_average_60m.csv', index=True, index_label='date')

Top_stocks_01m.to_csv(Path_Output+'Top_stocks_01m.csv', index=True, index_label='date')
Top_stocks_06m.to_csv(Path_Output+'Top_stocks_06m.csv', index=True, index_label='date')
Top_stocks_12m.to_csv(Path_Output+'Top_stocks_12m.csv', index=True, index_label='date')
Top_stocks_36m.to_csv(Path_Output+'Top_stocks_36m.csv', index=True, index_label='date')
Top_stocks_60m.to_csv(Path_Output+'Top_stocks_60m.csv', index=True, index_label='date')

Bottom_stocks_01m.to_csv(Path_Output+'Bottom_stocks_01m.csv', index=True, index_label='date')
Bottom_stocks_06m.to_csv(Path_Output+'Bottom_stocks_06m.csv', index=True, index_label='date')
Bottom_stocks_12m.to_csv(Path_Output+'Bottom_stocks_12m.csv', index=True, index_label='date')
Bottom_stocks_36m.to_csv(Path_Output+'Bottom_stocks_36m.csv', index=True, index_label='date')
Bottom_stocks_60m.to_csv(Path_Output+'Bottom_stocks_60m.csv', index=True, index_label='date')

