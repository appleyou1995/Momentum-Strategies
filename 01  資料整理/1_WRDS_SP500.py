from sqlalchemy import text

import pandas as pd
import numpy  as np
import os
import wrds


# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)

Path_Output = os.path.join(Path_dir, 'Code/01  輸出資料')


# %%  Setting query & Load data [ Common Stock ]

conn = wrds.Connection(wrds_username='irisyu')

libraries   = conn.list_libraries()
tables_crsp_a_indexes = conn.list_tables(library='crsp_a_indexes')

col_headers_msp500 = conn.describe_table(library='crsp_a_indexes', table='msp500')


# %%  Define the query for S&P 500 index data

query_sp500 = text("""
                   SELECT caldt, spindx
                   FROM crsp_a_indexes.msp500
                   WHERE caldt BETWEEN '1995-01-01' AND '2024-12-31'
                   """)

# Execute the query and fetch the data
df_SP500 = conn.raw_sql(query_sp500)


# %%  Compute log return & cumulative log return

# Set date as index and sort
df_SP500 = df_SP500.rename(columns={'caldt': 'date'})
df_SP500['date'] = pd.to_datetime(df_SP500['date'])
df_SP500 = df_SP500.set_index('date').sort_index()

# Compute next-period return using natural log: ln(S_{t+1} / S_t), stored at time t
df_SP500['SP500_next_period_return'] = np.log(df_SP500['spindx'].shift(-1) / df_SP500['spindx'])

# Compute cumulative return from t=0
df_SP500['SP500_cumulative_return'] = (1 + df_SP500['SP500_next_period_return']).cumprod() - 1

df_SP500.index = df_SP500.index.strftime('%Y-%m')


# %% Output

df_SP500.to_csv(Path_Output+'/SP500.csv', index=True)



