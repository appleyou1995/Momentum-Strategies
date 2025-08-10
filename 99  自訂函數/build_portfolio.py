import pandas as pd
import numpy  as np

# %%

def build_portfolio(df_max, dict_Top_Bottom, percentages):
    
    col_max = 'max_abs_column'
    col_val = 'original_value'

    # Column name
    out_cols = {p: f'portfolio_{int(p*100)}%' for p in percentages}

    # Calculate by month
    def calc_row(row):        
        ic_key = str(row[col_max]).split('_')[-1]
        tb = dict_Top_Bottom.get(ic_key)
        if tb is None or row.name not in tb.index:
            return {out_cols[p]: np.nan for p in percentages}

        sign = 1.0 if float(row[col_val]) >= 0 else -1.0
        # sign = +1 -> Top - Bottom
        # sign = -1 -> Bottom - Top
        
        res = {}
        for p in percentages:
            top = tb.loc[row.name, (p, 'top')]
            bot = tb.loc[row.name, (p, 'bottom')]
            res[out_cols[p]] = sign * (top - bot)
        return res

    add_df = df_max.apply(calc_row, axis=1, result_type='expand')
    return pd.concat([df_max, add_df], axis=1)