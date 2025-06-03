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
tables_crsp_a_stock = conn.list_tables(library='crsp_a_stock')

col_headers_msf = conn.describe_table(library='crsp_a_stock', table='msf')


# %%  Define the query for individual stock data

# Define the start and end years
year_start = 1995
year_end = 2024

# Initialize an empty DataFrame to store the results
df_msf_all = pd.DataFrame()

# Loop through each year and fetch data
for year in range(year_start, year_end + 1):
    print(f"Processing data for year: {year}")
    
    # Define the query for the specific year
    query_msf = text(f"""
                     SELECT 
                         date, permno, prc, hsiccd
                     FROM 
                         crsp_a_stock.msf
                     WHERE
                         date BETWEEN '{year}-01-01' AND '{year}-12-31'
                     AND NOT
                         ((hsiccd > 4900 AND hsiccd < 4999) OR (hsiccd > 6000 AND hsiccd < 6999))
                     """)
    
    # Execute the query and fetch the data for the year
    df_msf_year = conn.raw_sql(query_msf)
    
    # Append the result to the main DataFrame
    df_msf_all = pd.concat([df_msf_all, df_msf_year], ignore_index=True)

# Final DataFrame contains all years
print("Data fetching completed!")


# %%  Transform CRSP monthly stock price data into a firm × time matrix

# Pivot raw CRSP price data into a firm × date matrix
df_msf_pivot = pd.pivot_table(df_msf_all, index='permno', columns='date', values='prc')

# Remove firms with missing values in the latest available date column
df_msf_pivot = df_msf_pivot.dropna(subset=df_msf_pivot.columns[-1], how='any')

# Take absolute value of prices to handle CRSP's use of negative prices to indicate bid/ask quotes
df_msf_pivot = df_msf_pivot.abs()

# Convert column names (dates) from datetime to YYYY-MM string format
df_msf_pivot.columns = pd.to_datetime(df_msf_pivot.columns)
df_msf_pivot.columns = df_msf_pivot.columns.strftime('%Y-%m')

# Compute next-period log return: ln(P_{t+1} / P_t), stored at time t
df_next_period_return = np.log(df_msf_pivot.shift(-1, axis=1) / df_msf_pivot)


# %% Output

df_msf_pivot.to_csv(Path_Output+'/Individual_stock_price.csv', index=True)
df_next_period_return.to_csv(Path_Output+'/Individual_next_period_return.csv', index=True)










