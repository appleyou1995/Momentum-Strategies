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
