import pandas as pd

# %%

def filter_by_month_range(df, start, end):
    pidx = pd.PeriodIndex(pd.to_datetime(df.index).strftime('%Y-%m'), freq='M')
    mask = (pidx >= pd.Period(start, freq='M')) & (pidx <= pd.Period(end, freq='M'))
    out = df.loc[mask].copy()
    out.index = pidx[mask].strftime('%Y-%m')
    out = out.sort_index(ascending=True)
    return out