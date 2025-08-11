import pandas as pd
import numpy  as np

# %%

def portfolio_analysis(
    df_T: pd.DataFrame,
    annualize: bool,
    periods_per_year: int = 12,
    strategy_list=None,
    percentages=None,
    include_rows=('SP500'),
    meta_rows=('max_abs_column', 'max_abs_value', 'original_value'),
):
    """
    df_T:   index 為各項列（含策略列）、columns 為月份
    strategy_list: 例如 ['TB_BT','BuyT_BuyB','BuyT_SellB','SellB_SellT','BuyT','BuyB','SellT','SellB','BuyTSellB']
    percentages:    例如 [0.01, 0.05, 0.10]
    include_rows:   額外一定要包含的列（預設含 SP500）
    meta_rows:      應排除的資訊列
    """

    # --- 決定要分析的列名順序 ---
    row_order = []

    # 1) SP500
    if 'SP500' in include_rows and 'SP500' in df_T.index:
        row_order.append('SP500')

    # 2) 策略列（依 strategy_list × percentages 的順序）
    if strategy_list is not None and percentages is not None:
        for s in strategy_list:
            for p in percentages:
                name = f"{s}_{int(p*100)}%"
                if name in df_T.index:
                    row_order.append(name)
    else:
        # 自動抓策略列：以 % 結尾、且非 meta/include_rows
        auto_rows = [
            r for r in df_T.index
            if (r not in meta_rows) and (r not in include_rows) and r.endswith('%')
        ]
        auto_rows_sorted = sorted(auto_rows)
        row_order.extend(auto_rows_sorted)

    # 去重 + 過濾不存在者
    seen = set()
    row_order = [r for r in row_order if (r not in seen and not seen.add(r))]
    rows = [r for r in row_order if r in df_T.index]

    # --- 計算統計 ---
    vals = df_T.loc[rows].apply(pd.to_numeric, errors='coerce')

    means = vals.mean(axis=1)
    stds  = vals.std(axis=1, ddof=1)

    sharpe = means / stds

    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)

    # Outperform ratio：跟 SP500 比
    if 'SP500' in vals.index:
        sp = vals.loc['SP500']

        def _outperf(series):
            mask = series.notna() & sp.notna()
            if mask.sum() == 0:
                return np.nan
            return (series[mask] > sp[mask]).mean()

        outperform = vals.apply(_outperf, axis=1)
        outperform.loc['SP500'] = np.nan
    else:
        outperform = pd.Series(np.nan, index=vals.index)

    out = pd.DataFrame({
        'mean':   (means * 100).astype(float).round(2),
        'std':    (stds  * 100).astype(float).round(2),
        'sharpe': sharpe.astype(float).round(3),
        'outperform': outperform.astype(float).round(3)
    }).loc[rows]

    return out