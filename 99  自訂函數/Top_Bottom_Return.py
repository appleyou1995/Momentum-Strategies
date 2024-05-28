import pandas as pd


# %%

def Top_Bottom_Return(mom_rank, stock_price, log_return, num):
    
    mom_rank.index = pd.to_datetime(mom_rank.index).strftime('%Y-%m')

    # Top
    
    Top_stocks = pd.DataFrame(index=mom_rank.index, columns=range(1, num + 1))

    for date in mom_rank.index:
        
        stock_prices = stock_price.loc[date]
        
        # 過濾股價小於1的股票代碼
        filtered_stocks = stock_prices[stock_prices > 1].index.astype(str)
        
        # 如果沒有符合條件的股票，則跳過當月份
        if len(filtered_stocks) == 0:
            continue
        
        rank = mom_rank.loc[date]    
        Top_ranked_stocks = rank[rank.index.isin(filtered_stocks)].nlargest(num).index    
        Top_stocks.loc[date, :len(Top_ranked_stocks)] = Top_ranked_stocks


    # Bottom

    Bottom_stocks = pd.DataFrame(index=mom_rank.index, columns=range(1, num + 1))

    for date in mom_rank.index:
        
        stock_prices = stock_price.loc[date]
        
        # 過濾股價小於1的股票代碼
        filtered_stocks = stock_prices[stock_prices > 1].index.astype(str)
        
        # 如果沒有符合條件的股票，則跳過當月份
        if len(filtered_stocks) == 0:
            continue
        
        rank = mom_rank.loc[date]    
        Bottom_ranked_stocks = rank[rank.index.isin(filtered_stocks)].nsmallest(num).index
        Bottom_stocks.loc[date, :len(Bottom_ranked_stocks)] = Bottom_ranked_stocks


    # log return

    Top_logreturn = pd.DataFrame(columns=Top_stocks.columns)

    for time in Top_stocks.index:
        Top_stock_codes = Top_stocks.loc[time].values.astype(int)
        Top_logreturn.loc[time] = log_return.loc[time, Top_stock_codes].values

    Top_average = Top_logreturn.mean(axis=1).to_frame(name='Top_average')


    Bottom_logreturn = pd.DataFrame(columns=Bottom_stocks.columns)

    for time in Bottom_stocks.index:
        Bottom_stock_codes = Bottom_stocks.loc[time].values.astype(int)
        Bottom_logreturn.loc[time] = log_return.loc[time, Bottom_stock_codes].values

    Bottom_average = Bottom_logreturn.mean(axis=1).to_frame(name='Bottom_average')


    Top_minus_Bottom = Top_average['Top_average'] - Bottom_average['Bottom_average']
    Top_minus_Bottom = Top_minus_Bottom.to_frame(name='Top_minus_Bottom')

    Bottom_minus_Top = Bottom_average['Bottom_average'] - Top_average['Top_average']
    Bottom_minus_Top = Bottom_minus_Top.to_frame(name='Bottom_minus_Top')

    Buy_Top     =  Top_average
    Sell_Top    = -Top_average
    Buy_Bottom  =  Bottom_average
    Sell_Bottom = -Bottom_average

    result = pd.concat([Top_minus_Bottom, Bottom_minus_Top, 
                        Buy_Top, Sell_Top, Buy_Bottom, Sell_Bottom], 
                       axis=1)

    result.columns = ['Top_minus_Bottom', 'Bottom_minus_Top', 
                      'Buy_Top', 'Sell_Top', 'Buy_Bottom', 'Sell_Bottom']
    
    return result, Top_average, Bottom_average, Top_stocks, Bottom_stocks