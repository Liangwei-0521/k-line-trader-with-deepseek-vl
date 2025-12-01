import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import json
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from textwrap import dedent
from llm.v_llm import initial_llm
from llm.prompts import trading_prompt
from tmp.handle import transform_response
from account.stock_account import StockAccount
from picture.plot import plot_kline
from data.base_plot import prepare_data, rolling



class TextMemory:
    def __init__(self, max_len=10):
        self.max_len = max_len
        self.memories = []

    def add(self, text: str):
        # Only text, no images
        self.memories.append(text)
        if len(self.memories) > self.max_len:
            self.memories.pop(0)

    def get_all(self):
        return self.memories


def run( 
        model_name:str,
        model_path: str,
        root_path:str,
        country:str,
        symbol:str,
        start_date:str,
        end_date:str,
        win_len:int,
        args=None,
        model=None,
    ):
    print("🚀 Starting QuantBot...")

    # llm = initial_llm(model_path)
    llm = model

    trading_record = []

    # stock data list
    df = prepare_data(root_path=root_path, country=country, symbol=symbol,
                      start_date=start_date, end_date=end_date)

    df_list = rolling(win_len=win_len, df=df)

    print("🤖 QuantBot is ready to assist you!")
    quantbot_account = StockAccount(num_stock=1)
    quantbot_account.initial_account()

    for idx in range(0, len(df_list)-1):
        print('====== 🔰 Trading index ====== :', idx)
        if idx == 0:
            quant_bot_account = dedent(
            f"""
                target transaction date is: {df_list[idx + 1].index[-1].strftime('%Y-%m-%d')}\n
                For the first transaction, the cumulative rate of return is temporarily 0.00, and the current yield of each stock is temporarily 0 \n
            """)
        else:
            quant_bot_account = dedent(
            f"""
                Target transaction date is: {df_list[idx + 1].index[-1].strftime('%Y-%m-%d')}\n
                After a series of investment decisions, the current cumulative rate of return is: {account_info["profit"] * 100 } % ;\n
                The transaction cost rate: 0.3% ,\n
                The current holding shares of is: {account_info["holdings"]} ,\n
                The current account cash of is: {account_info["balance"]},
            """)

        # create k chat
        plot_kline(
            data_list=df_list, 
            country=country,
            symbol=symbol,
            plot_idx=idx,
            root_save_path = f'./picture/show/',
        )


        response = llm.run(
            user_query=quant_bot_account, 
            image_path=f'./picture/show/{country}/{symbol}/kline_{idx}.png',
            args=args)
        
        print('Raw response from LLM: \n', response)
        
        # 返回json数据结构
        try:
            decison, decision_content = transform_response(content=response, return_dict=True)
            
        except Exception as e:
            print('Error in transforming response: ', e) 
            # default hold decision 
            decision_content = { "trading date": f"{df_list[idx + 1].index[-1].strftime('%Y-%m-%d')}", 
                                "decision reason": "The stock is showing a downward trend with no clear reversal signals. The MACD is in the negative zone and the RSI is below 30, indicating oversold conditions. However, the stock price is still below the 200-day moving average, suggesting a potential long-term downtrend. The risk-reward ratio is not favorable at the moment.", 
                                "trading decision": 0}
        
        trading_record.append(decision_content)

        # save the file 
        dir_path = os.path.join('./result/'+country+'/', model_name)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f'quantbot_{country}_{symbol}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trading_record, f, ensure_ascii=False, indent=4)

        print('🤖QuantBot decison '+str(idx)+': \n ', decision_content)

        # account info 
        open_price_val = df_list[idx + 1]['Open'].iloc[-1]
        close_price_val = df_list[idx + 1]['Close'].iloc[-1]
        account_info = quantbot_account.update_account(
            # action type: int
            action=decision_content["trading decision"],
            # the open price of target trading date
            open_price=open_price_val,
            # the close price of target trading date
            close_price=close_price_val,
        )

        print('account information: \n ', account_info)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()

    
    run(
        model = initial_llm("./models/deepseek-vl2-small"),
        model_path ="./models/deepseek-vl2-small",
        model_name='deepseek-vl2-small',
        root_path='./data/stock_list/',
        country='us',
        symbol='BA',
        start_date='2020-01-01',
        end_date='2020-12-31',
        win_len=20,
        args=args
    )