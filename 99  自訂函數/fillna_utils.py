def fillna_with_column_median(df):
    for column in df.columns:
        column_median = df[column].median()
        df[column].fillna(column_median, inplace=True)
    return df


def fillna_with_column_mean(df):
    for column in df.columns:
        column_mean = df[column].mean()
        df[column].fillna(column_mean, inplace=True)
    return df


def fillna_with_row_mean(df):
    row_mean = df[df.notnull()].mean()
    df.fillna(row_mean, inplace=True)
    return df
