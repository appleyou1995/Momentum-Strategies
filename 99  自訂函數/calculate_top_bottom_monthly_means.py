import numpy as np
import pandas as pd


# %%

def calculate_top_bottom_monthly_means(log_return_tradable: pd.DataFrame,
                                       mom: pd.DataFrame,
                                       percentages) -> pd.DataFrame:
    """
    依據 mom 的排序，計算每月各百分比的 top/bottom log_return_tradable 平均值與檔數。
    - percentages: 可為單一數值 (如 0.1) 或 list/tuple (如 [0.01, 0.05, 0.1])
                   表示比例（0~1）。
    回傳: DataFrame，index 與輸入相同，多層欄位 (percentage, ['top','bottom','count'])。
    """
    # 轉成 list 形式
    if not isinstance(percentages, (list, tuple)):
        percentages = [percentages]
    
    # 對齊形狀
    mom = mom.reindex_like(log_return_tradable)

    all_results = {}
    for p in percentages:
        results = []
        for dt in mom.index:
            m = mom.loc[dt].dropna()
            n = len(m)
            if n == 0:
                results.append((np.nan, np.nan, 0))
                continue

            k = max(int(np.floor(n * p)), 1)

            top_ids = m.nlargest(k).index
            bot_ids = m.nsmallest(k).index

            lr = log_return_tradable.loc[dt]
            top_mean = lr[top_ids].mean(skipna=True)
            bot_mean = lr[bot_ids].mean(skipna=True)

            results.append((top_mean, bot_mean, k))

        all_results[p] = pd.DataFrame(results, index=mom.index,
                                      columns=['top', 'bottom', 'count'])
    
    # 合併成多層欄位 DataFrame
    out = pd.concat(all_results, axis=1)
    out.columns.names = ['percentage', 'metric']
    return out
