import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from data.base_plot import prepare_data, rolling


def plot_kline(
    data_list:list, 
    country='us',
    symbol='BA',
    plot_idx=100,
    root_save_path = None,
    ):
    """专业证券风格 K 线图（含 MA + 成交量 + MACD) """

    # 选择最后一个窗口的 plot_length 天
    data = data_list[plot_idx].copy()

    # === 3. 设置颜色和样式 ===
    market_color = mpf.make_marketcolors(
        up='#e74c3c', down='#2ca02c',
        edge='inherit', wick='inherit',
        volume='inherit'
    )
    style = mpf.make_mpf_style(
        marketcolors=market_color,
        figcolor='white',
        gridcolor='#dcdcdc',
        gridstyle='--'
    )

    # === 4. 创建画布 ===
    start_point = data.index[0].strftime('%b-%d')  # e.g., 'May-01'
    end_point = data.index[-1].strftime('%b-%d') 

    fig = mpf.figure(style=style, figsize=(10, 7.5), facecolor='white')

    fig.suptitle(
        f"Stock — Price Trend ({start_point} ~ {end_point})",
        fontsize=14, weight="bold", y=0.93
    )

    ax_price = fig.add_axes([0.06, 0.28, 0.88, 0.60])
    ax_volume = fig.add_axes([0.06, 0.18, 0.88, 0.10], sharex=ax_price)
    ax_macd = fig.add_axes([0.06, 0.05, 0.88, 0.13], sharex=ax_price)

    ax_price.set_ylabel('Price')
    ax_volume.set_ylabel('Volume')
    ax_macd.set_ylabel('MACD')

    # === 5. 添加附加图层（MA + MACD） ===
    ap = []
    # 均线
    ap.append(mpf.make_addplot(data['MA5'],  ax=ax_price, color='#f39c12', width=2.0, label='MA5'))
    ap.append(mpf.make_addplot(data['MA10'], ax=ax_price, color='#3498db', width=2.0, label='MA10'))
    ap.append(mpf.make_addplot(data['MA20'], ax=ax_price, color='#9b59b6', width=2.0, label='MA20'))
    # ap.append(mpf.make_addplot(data['MA30'], ax=ax_price, color='#e84393', width=2.0, label='MA30'))

    # MACD
    ap.append(mpf.make_addplot(data['DEA'],  ax=ax_macd, color='#e67e22', width=2.0, label='DEA'))
    ap.append(mpf.make_addplot(data['MACD'], ax=ax_macd, color='#2980b9', width=2.0, label='MACD'))

    # MACD 柱状图
    bar_r = np.where(data['MACD'] > 0, data['MACD'], 0)
    bar_g = np.where(data['MACD'] <= 0, data['MACD'], 0)
    ap.append(mpf.make_addplot(bar_r, type='bar', color="#e9230d", ax=ax_macd, alpha=0.7))
    ap.append(mpf.make_addplot(bar_g, type='bar', color="#056b05", ax=ax_macd, alpha=0.7))

    # === 6. 绘制图形 ===
    mpf.plot(
        data,
        ax=ax_price,
        volume=ax_volume,
        addplot=ap,
        type='candle',
        style=style,
        xrotation=0,
        returnfig=True
    )

    # === 7. 显示图例 ===
    ax_price.legend(loc='upper left', fontsize=9)
    ax_macd.legend(loc='upper left', fontsize=9)

    # 保持图片
    path = os.path.join(root_save_path, f'{country}/{symbol}/')
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f'kline_{plot_idx}.png')
    fig.savefig(file_path, dpi=300)


if __name__ == '__main__':

    df = prepare_data(
        root_path='./data/stock_list/',
        country='us',
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2020-12-31'
    )

    df_list = rolling(win_len=20, df=df)
    
    plot_kline(
        data_list = df_list, 
        country='us',
        symbol='AAPL',
        plot_idx=0,
        root_save_path = f'./picture/show/',
    )
    # print(df_list[-1].index[0].strftime('%Y-%m-%d'), df_list[-1].index[-1].strftime('%Y-%m-%d'))
    print(df_list[0]['Close'].iloc[-1])
    print(df_list[0]['Open'].iloc[-1])
    
  