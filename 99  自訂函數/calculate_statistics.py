def calculate_statistics(df):
    stats = {
        'N': df.shape[0],
        'Minimum': df.min().min(),
        '25%': df.quantile(0.25).min(),
        '50%': df.median().min(),
        '75%': df.quantile(0.75).min(),
        'Maximum': df.max().max(),
        'Mean': df.mean().mean(),
        'SD': df.std().mean()
    }
    return stats
