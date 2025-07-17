import pandas as pd
import os
import matplotlib.pyplot as plt


# %%  論文資料夾路徑

Path_PaperFolder = '我的雲端硬碟/學術｜研究與論文/論文著作/動能因子與機器學習'


# %%  Win 資料夾路徑

Path_Win = 'D:/Google/'
Path_dir = os.path.join(Path_Win, Path_PaperFolder)

Path_Output = os.path.join(Path_dir, 'Code/01  輸出資料')


# %%  Import data

file_path = r"D:\Google\我的雲端硬碟\學術｜研究與論文\論文著作\動能因子與機器學習\Data\CRSP_1957_2024.csv"
df = pd.read_csv(file_path)


# %%  

df_pivot = pd.pivot_table(df, index='PERMNO', columns='date', values='PRC')
df_pivot = df_pivot.abs()


# %%  

# 找出每支股票中有非 NaN 的期間
valid_range = df_pivot.notna()

# 找出每支股票連續期間的起訖點
first_valid = valid_range.idxmax(axis=1)                                       # 每 row 的第一個 True 對應的欄位
last_valid = valid_range.iloc[:, ::-1].idxmax(axis=1)                          # 每 row 的最後一個 True 對應的欄位

# Get start and end column positions (integers)
start_idx = df_pivot.columns.get_indexer(first_valid)
end_idx = df_pivot.columns.get_indexer(last_valid)

# Check for interruptions between first and last valid dates
interrupted = []

for i, stock in enumerate(df_pivot.index):
    row = df_pivot.loc[stock]
    # Get values between first and last valid column
    values = row.iloc[start_idx[i]:end_idx[i] + 1]  # include end
    interrupted.append(values.isna().any())

# Convert to Series with index
interrupted_series = pd.Series(interrupted, index=df_pivot.index, name='Has_Interruption')

valid_periods = pd.DataFrame({
    'First_Valid_Date': first_valid,
    'Last_Valid_Date': last_valid,
    'Has_Interruption': interrupted_series
})

# Get how many are interrupted
print(f"Number of stocks with price interruptions: {interrupted_series.sum()}")


# %%

interrupted_permnos = interrupted_series[interrupted_series].index.tolist()
df_interrupted = df_pivot.loc[interrupted_permnos]
df_check = valid_periods[valid_periods['Has_Interruption']]

# 有一大段時間都不見 ⭢ 放著不管？
df_10007 = df[df['PERMNO'] == 10007]
df_10012 = df[df['PERMNO'] == 10012]
df_10028 = df[df['PERMNO'] == 10028]
df_10051 = df[df['PERMNO'] == 10051]

# 其中一個月不見 ⭢ 拿前後月份平均？
df_10021 = df[df['PERMNO'] == 10021]
df_10647 = df[df['PERMNO'] == 10647]
df_10050 = df[df['PERMNO'] == 10050]


# %%  The number of stocks & the average number of stocks per month

monthly_stock_counts = df_pivot.notna().sum(axis=0)

monthly_stock_counts.plot(figsize=(12, 8), title='Number of Stocks with Prices per Month')
plt.xlabel('Date')
plt.ylabel('Number of Stocks')
plt.tight_layout()
plt.show()

# Calculate the average number of stocks per month
average_stocks_per_month = monthly_stock_counts.mean()
print(f"Average number of stocks per month: {average_stocks_per_month:.2f}")


# %%  1957-2016 check

# (1) "The number of stocks in our sample is almost 30,000"

# 先確保 df['date'] 是 datetime 格式
df['date'] = pd.to_datetime(df['date'])

# 篩選出 1957-03 到 2016-12 的資料
df_in_sample = df[(df['date'] >= '1957-03-01') & (df['date'] <= '2016-12-31')]

# 取得不重複的 PERMNO 數量
num_unique_stocks = df_in_sample['PERMNO'].nunique()
print(f"(1) Number of unique stocks from 1957-03 to 2016-12: {num_unique_stocks}")


# (2) "with the average number of stocks per month exceeding 6,200"

# 將欄位（即月份）轉為 datetime 格式
monthly_stock_counts.index = pd.to_datetime(monthly_stock_counts.index)

# 篩選出 1957-03 到 2016-12 的區間
mask = (monthly_stock_counts.index >= '1957-03-01') & (monthly_stock_counts.index <= '2016-12-31')
monthly_in_range = monthly_stock_counts[mask]

# 重新計算平均
average_1957_2016 = monthly_in_range.mean()
print(f"(2) Average number of stocks per month (1957–2016): {average_1957_2016:.2f}")


# %%  Next-period log return

# Compute next-period log return: ln(P_{t+1} / P_t), stored at time t
df_next_period_return = np.log(df_msf_pivot.shift(-1, axis=1) / df_msf_pivot)


# %%  Output

df_pivot.to_csv(Path_Output+'/Individual_stock_price_manual.csv', index=True)
