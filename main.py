from bot import run
from argparse import ArgumentParser
from bot import run 
from tqdm import tqdm


import os
import numpy as np
import pandas as pd
from llm.v_llm import initial_llm

def calc_cv_for_symbol(file_path: str, start_time: str, end_time: str):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_time) & (df['Date'] <= end_time)
    next_df = df.loc[mask].copy()
    
    if next_df.empty:  # 没有数据就返回 None
        return None
    
    mean_close = next_df['Close'].mean()
    if mean_close == 0:  # 避免除零错误
        return None
    return next_df['Close'].std() / mean_close

def batch_calc_cv(root_path: str, start_time: str, end_time: str):
    results = []
    for filename in os.listdir(root_path):
        if filename.endswith('.csv'):  # 只处理 CSV 文件
            file_path = os.path.join(root_path, filename)
            cv = calc_cv_for_symbol(file_path, start_time, end_time)
            if cv is not None:
                results.append({'symbol': filename.replace('.csv', ''), 'cv': cv})
    
    # 转成 DataFrame，方便排序
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by='cv', ascending=False).reset_index(drop=True)
    return df_result



if __name__ == "__main__":
    
    country_list = ['us', 'cn']

    model_name = 'deepseek-vl2-small'
    model_path ="./models/deepseek-vl2-small"
    
    initial_model = initial_llm("./models/deepseek-vl2-small")

    parser = ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=-1,
                                help="chunk size for the model for prefiiling. "
                                        "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                                        "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()


    for item in country_list:
        country = item
        model_name = model_name
        root_path = f'./data/stock_list/{country}/next/'

        start_time = '2020-01-01'
        end_time = '2020-12-31'
        cv_df = batch_calc_cv(root_path, start_time, end_time)

        for ticker in tqdm(cv_df['symbol'][:]):
            if country == 'cn':
                item = [item.split('.')[0].split('_')[-1] + '_ss' for item in os.listdir('./result/'+country+'/'+model_name)]
                
            if country == 'us':
                item = [item.split('.')[0].split('_')[-1] for item in os.listdir('./result/'+country+'/'+model_name)]

            if ticker in item:
                pass
            else:
                if country == 'cn':
                    ticker = ticker.split('_')[0]
                    print('Running for:', ticker)

            run(
                model=initial_model, 
                model_path=model_path,
                model_name=model_name,
                root_path='./data/stock_list/',
                country=country,
                symbol=ticker,
                start_date='2020-01-01',
                end_date='2020-12-31',
                win_len=20,
                args=args
            )