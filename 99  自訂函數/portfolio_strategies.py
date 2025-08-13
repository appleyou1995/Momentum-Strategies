import pandas as pd
import numpy  as np

# %%

# -------------------------------------------------------------------------
# 定義每種策略計算方式（輸入 sign, top, bottom）
# -------------------------------------------------------------------------
def _TB_BT(sign, top, bottom):
    return sign * (top - bottom)

def _BuyT_BuyB(sign, top, bottom):
    return top if sign >= 0 else bottom

def _BuyT_SellB(sign, top, bottom):
    return top if sign >= 0 else -bottom

def _SellB_SellT(sign, top, bottom):
    return -bottom if sign >= 0 else -top

def _BuyT(sign, top, bottom):
    return top

def _BuyB(sign, top, bottom):
    return bottom

def _SellT(sign, top, bottom):
    return -top

def _SellB(sign, top, bottom):
    return -bottom

def _BuyTSellB(sign, top, bottom):
    return top - bottom


# -------------------------------------------------------------------------
# 策略名稱與對應函數的映射
# -------------------------------------------------------------------------
STRATEGY_FUNCS = {
    'TB_BT': _TB_BT,
    'BuyT_BuyB': _BuyT_BuyB,
    'BuyT_SellB': _BuyT_SellB,
    'SellB_SellT': _SellB_SellT,
    'BuyT': _BuyT,
    'BuyB': _BuyB,
    'SellT': _SellT,
    'SellB': _SellB,
    'BuyTSellB': _BuyTSellB,
}


# -------------------------------------------------------------------------
# 單一策略計算
# -------------------------------------------------------------------------
def build_portfolio(df_max, dict_Top_Bottom, percentages, strategy_name):
    """
    df_max 需要有：
      - 'max_abs_column'：用來拆出 ic_key（你原本以 '_' 切最後一段）
      - 'original_value'：用來判斷 sign（>=0 視為正）

    dict_Top_Bottom[ic_key]：
        DataFrame，index=日期，columns=MultiIndex (p, 'top'/'bottom')

    strategy_name：如 'TB_BT', 'BuyT_BuyB', 'SellT' 等
    """
    if strategy_name not in STRATEGY_FUNCS:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")

    col_max = 'max_abs_column'
    col_val = 'original_value'

    def _colname(p):
        return f"{strategy_name}_{int(p*100)}%"

    out_cols = {p: _colname(p) for p in percentages}
    func = STRATEGY_FUNCS[strategy_name]

    def calc_row(row):
        ic_key = str(row[col_max]).split('_')[-1]
        tb = dict_Top_Bottom.get(ic_key)
        if tb is None or row.name not in tb.index:
            return {out_cols[p]: np.nan for p in percentages}

        sign = 1.0 if float(row[col_val]) >= 0 else -1.0

        res = {}
        for p in percentages:
            top = tb.loc[row.name].get((p, 'top'), np.nan)
            bottom = tb.loc[row.name].get((p, 'bottom'), np.nan)
            res[out_cols[p]] = func(sign, top, bottom)
        return res

    add_df = df_max.apply(calc_row, axis=1, result_type='expand')
    return pd.concat([df_max, add_df], axis=1)


# -------------------------------------------------------------------------
# 多策略一次計算
# -------------------------------------------------------------------------
def build_strategies(df_max, dict_Top_Bottom, percentages, strategy_list):
    """
    strategy_list: e.g. ['TB_BT', 'BuyT_BuyB', 'SellT']
    """
    out = df_max.copy()
    for strategy_name in strategy_list:
        out = build_portfolio(out, dict_Top_Bottom, percentages, strategy_name)
    return out


# -------------------------------------------------------------------------
# 計算累積報酬率
# -------------------------------------------------------------------------
def build_cumulative_return(
    df: pd.DataFrame,
    strategy_plot,
    sp500_name: str = "SP500",
    exclude_rows=("max_abs_column","max_abs_value","original_value","tradable"),
    prefix_match: bool = True,
) -> pd.DataFrame:
    """
    df:  列=策略名稱、欄=月份(字串 'YYYY-MM' 或 Period)，元素=月度對數報酬
    strategy_plot: 例如 ['TB_BT'] 或 ['TB_BT','BuyT_SellB']
    sp500_name: SP500 那一列在 df 的名稱
    exclude_rows: 要排除的資訊列
    prefix_match: True 時用前綴比對 (e.g., 'TB_BT' 會抓到 'TB_BT_1%/5%/10%')
                  False 時僅比對完整列名
    回傳：只含所選策略+SP500 的「累積對數報酬」DataFrame（**往左相加**）
    """
    # 1) 列篩選
    want = set()
    for idx in df.index:
        if idx in exclude_rows:
            continue
        if idx == sp500_name:
            want.add(idx); continue
        for s in strategy_plot:
            if prefix_match:
                if idx.startswith(s + "_") or idx == s:
                    want.add(idx)
            else:
                if idx == s:
                    want.add(idx)

    sub = df.loc[[idx for idx in df.index if idx in want]].copy()

    # 2) 確保為數值
    sub = sub.apply(pd.to_numeric, errors="coerce")

    # 3) 累積對數報酬
    cumlog = sub.cumsum(axis=1)

    return cumlog


# -------------------------------------------------------------------------
# 計算再投資報酬率
# -------------------------------------------------------------------------
def build_reinvested_return(
    df: pd.DataFrame,
    strategy_plot,
    sp500_name: str = "SP500",
    exclude_rows=("max_abs_column","max_abs_value","original_value","tradable"),
    prefix_match: bool = True,
    start_value: float = 1.0,
    return_mode: str = "acc_ret",    # 'acc_ret' | 'wealth'
) -> pd.DataFrame:
    """
    依策略前綴挑列後，計算「再投資」結果（輸入已為對數報酬）。

    df:            列=策略/基準，欄=月份('YYYY-MM' 或 Period)，元素=月度「對數報酬」
    strategy_plot: 例如 ['TB_BT'] 或 ['TB_BT','BuyT_SellB']
    sp500_name:    基準列名（若有）
    exclude_rows:  要排除的資訊列
    prefix_match:  True=用前綴比對（'TB_BT' 抓到 'TB_BT_1%/5%/10%'）
    start_value:   初始淨值
    return_mode:   回傳型態：
                   - 'acc_ret'：累積簡單報酬（從 0 起）
                   - 'wealth'：淨值曲線（從 start_value 起）
                   - 'both'：回傳 (acc_ret_df, wealth_df)
    """
    # 1) 列篩選
    want = set()
    for idx in df.index:
        if idx in exclude_rows:
            continue
        if idx == sp500_name:
            want.add(idx); continue
        for s in strategy_plot:
            if (prefix_match and (idx.startswith(s + "_") or idx == s)) or ((not prefix_match) and idx == s):
                want.add(idx); break
    sub = df.loc[[idx for idx in df.index if idx in want]].copy()

    # 2) 轉數值
    sub = sub.apply(pd.to_numeric, errors="coerce")

    # 3) 再投資累積報酬
    acc_log = sub.cumsum(axis=1)
    wealth  = start_value * np.exp(acc_log)
    acc_ret = np.expm1(acc_log)

    return wealth if return_mode == "wealth" else acc_ret
