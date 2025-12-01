import os
import numpy as np
import pandas as pd 


def prepare_data(root_path:str, country:str, symbol:str, start_date:str, end_date:str):

    # 1、select the 
    df_path = os.path.join(root_path, country, "next", f"{symbol}.csv")
    df = pd.read_csv(df_path)

    df['Date'] = pd.to_datetime(df['Date'])
    cols_to_drop = ["Bbands_Upper", "Bbands_Middle", "Bbands_Lower"]
    df = df.drop(columns=cols_to_drop)
                 
    # 2、get the time sequence based on time tag
    df['Date'].dt.strftime("%Y-%m-%d")
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    next_df = df.loc[mask].copy().reset_index(drop=True)
    next_df['Date'] = pd.to_datetime(next_df['Date'])
    next_df.set_index('Date', inplace=True)

    return next_df


def rolling(win_len:int, df:pd.DataFrame):
    """
    params:
        win_len: 
    """
    df_list = []
    for idx in range(win_len, len(df)+1):
        df_list.append(df[idx-win_len:idx])
         
    return df_list



if __name__ == '__main__':
    df = prepare_data(
        root_path='./data/stock_list/',
        country='us',
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2020-12-31'
    )

    df_list = rolling(win_len=10, df=df)
    print(df_list[1].index[0].strftime('%b-%d'), df_list[-1].index[-1].strftime('%b-%d'))
    print(df_list[1].index[0].strftime('%yy-%m-%d'), df_list[-1].index[-1].strftime('%b-%d'))
