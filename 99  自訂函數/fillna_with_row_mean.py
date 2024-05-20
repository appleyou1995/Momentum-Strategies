def fillna_with_row_mean(row):
    row_mean = row[row.notnull()].mean()
    row.fillna(row_mean, inplace=True)
    return row