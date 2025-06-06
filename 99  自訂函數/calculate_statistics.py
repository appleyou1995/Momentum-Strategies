import pandas as pd

def calculate_statistics(df):
    stats = {}
    for col in df.columns:
        stats[col] = {
            'N': df[col].shape[0],
            'Minimum': df[col].min(),
            '25%': df[col].quantile(0.25),
            '50%': df[col].median(),
            '75%': df[col].quantile(0.75),
            'Maximum': df[col].max(),
            'Mean': df[col].mean(),
            'SD': df[col].std()
        }
    return pd.DataFrame(stats)
