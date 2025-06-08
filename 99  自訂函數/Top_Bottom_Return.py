import pandas as pd
import numpy  as np


# %%

def Top_Bottom_Return(mom_rank, stock_price, log_return, pct):
    
    mom_rank.index = pd.to_datetime(mom_rank.index).strftime('%Y-%m')

    # Top
    Top_stocks = pd.DataFrame(index=mom_rank.index)

    for date in mom_rank.index:
        
        stock_prices = stock_price.loc[date]
        
        # 過濾股價小於1的股票代碼
        filtered_stocks = stock_prices[stock_prices > 1].index.astype(str)
        
        if len(filtered_stocks) == 0:
            continue
        
        # 動態決定 Top 選幾支
        num_top = max(1, int(len(filtered_stocks) * pct))    # 至少選 1 支
        
        rank = mom_rank.loc[date]
        Top_ranked_stocks = rank[rank.index.isin(filtered_stocks)].nlargest(num_top).index    
        Top_stocks.loc[date, 'Top_stocks'] = ','.join(Top_ranked_stocks)   # 儲存為逗號分隔字串
        
    # Bottom
    Bottom_stocks = pd.DataFrame(index=mom_rank.index)

    for date in mom_rank.index:
        
        stock_prices = stock_price.loc[date]
        
        filtered_stocks = stock_prices[stock_prices > 1].index.astype(str)
        
        if len(filtered_stocks) == 0:
            continue
        
        num_bottom = max(1, int(len(filtered_stocks) * pct))    # 至少選 1 支
        
        rank = mom_rank.loc[date]    
        Bottom_ranked_stocks = rank[rank.index.isin(filtered_stocks)].nsmallest(num_bottom).index
        Bottom_stocks.loc[date, 'Bottom_stocks'] = ','.join(Bottom_ranked_stocks)
        
    # log return
    Top_average = []
    Bottom_average = []

    for time in Top_stocks.index:
        
        if pd.isna(Top_stocks.loc[time, 'Top_stocks']):
            Top_average.append(np.nan)
            continue
        
        Top_stock_codes = [int(code) for code in Top_stocks.loc[time, 'Top_stocks'].split(',')]
        returns = log_return.loc[time, Top_stock_codes].values
        Top_average.append(np.nanmean(returns))
        
    for time in Bottom_stocks.index:
        
        if pd.isna(Bottom_stocks.loc[time, 'Bottom_stocks']):
            Bottom_average.append(np.nan)
            continue
        
        Bottom_stock_codes = [int(code) for code in Bottom_stocks.loc[time, 'Bottom_stocks'].split(',')]
        returns = log_return.loc[time, Bottom_stock_codes].values
        Bottom_average.append(np.nanmean(returns))
        
    Top_average = pd.Series(Top_average, index=Top_stocks.index, name='Top_average')
    Bottom_average = pd.Series(Bottom_average, index=Bottom_stocks.index, name='Bottom_average')
    
    # 計算 result 各種欄位
    Top_minus_Bottom = Top_average - Bottom_average
    Bottom_minus_Top = Bottom_average - Top_average
    Buy_Top     =  Top_average
    Sell_Top    = -Top_average
    Buy_Bottom  =  Bottom_average
    Sell_Bottom = -Bottom_average

    result = pd.concat([Top_minus_Bottom.to_frame(name='Top_minus_Bottom'),
                        Bottom_minus_Top.to_frame(name='Bottom_minus_Top'),
                        Buy_Top.to_frame(name='Buy_Top'),
                        Sell_Top.to_frame(name='Sell_Top'),
                        Buy_Bottom.to_frame(name='Buy_Bottom'),
                        Sell_Bottom.to_frame(name='Sell_Bottom')], axis=1)
    
    return result, Top_average.to_frame(), Bottom_average.to_frame(), Top_stocks, Bottom_stocks
