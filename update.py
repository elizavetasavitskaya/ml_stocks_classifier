import datetime
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from stockstats import wrap

warnings.filterwarnings("ignore")

INDICATORS = ['close', 'open', 'high', 'low', 'volume', 'rsi', 'stochrsi', 'close_50_sma', 'close_200_sma', 'boll_ub',
              'boll_lb',
              'macd', 'macds', 'wr', 'atr', 'dx', 'pdi', 'ndi', 'ao']


def get_json_parsed_data(url):
    """Return json object from https://financialmodelingprep.com/ url"""
    return json.loads(urlopen(url).read().decode('utf-8'))


def load_stocks_list(exchange):
    traded_stocks_url = ('https://financialmodelingprep.com/api/v3/available-traded/list?apikey'
                         '=1401249d9cdf8f975028aa69f2ef15c7')
    stocks = pd.DataFrame(get_json_parsed_data(traded_stocks_url))
    stocks = stocks[(stocks.exchangeShortName == exchange) & (stocks.type == 'stock')].symbol.tolist()
    return stocks


def create_indicators_matrix(exchange, indicators, number_of_treads):
    """Loading close/high/low/volume for specified exchange from provider and
  convert them into indicators matrix using stockstats"""
    traded_stocks__url = 'https://financialmodelingprep.com/api/v3/available-traded/list?apikey=1401249d9cdf8f975028aa69f2ef15c7'
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365 * 2)
    period = 250

    def load_close_prices_and_volumes(stock):
        try:
            price_url = ('https://financialmodelingprep.com/api/v4/historical-price-adjusted'
                         f'/{stock}/1/day/{start}/{end}?apikey=1401249d9cdf8f975028aa69f2ef15c7')
            data = pd.DataFrame(get_json_parsed_data(price_url)['results'])
            close, open, high, low, volume = data[['c']], data[['o']], data[['h']], data[['l']], data[['v']]
            close.fillna(method='ffill', inplace=True)
            high.fillna(method='ffill', inplace=True)
            low.fillna(method='ffill', inplace=True)
            volume.fillna(method='ffill', inplace=True)
            if len(close) >= period and volume['v'][0] >= 20000:
                close = np.flip(close['c'].values[:period])
                open = np.flip(open['o'].values[:period])
                high = np.flip(high['h'].values[:period])
                low = np.flip(low['l'].values[:period])
                volume = np.flip(volume['v'].values[:period])
                return [stock, exchange, {'close': close, 'open': open, 'high': high, 'low': low, 'volume': volume}]
        except (KeyError, IndexError, TypeError, URLError) as error:
            return None
        return None

    matrix_of_indicators = list()
    stocks = pd.DataFrame(get_json_parsed_data(traded_stocks__url))
    stocks = stocks[(stocks.exchangeShortName == exchange) & (stocks.type == 'stock')].symbol.tolist()
    print(f'Number of stocks at {exchange}:', len(stocks))
    print('Loading Data ...')
    with ThreadPoolExecutor(max_workers=number_of_treads) as executor:
        for data in executor.map(load_close_prices_and_volumes, stocks):
            try:
                if data:
                    scaler_c = Scaler()
                    matrix_of_indicators.append([data[0], data[1], wrap(pd.DataFrame(data[2]))[indicators],
                                                 scaler_c.fit_transform(
                                                     TimeSeries.from_values(data[2]['close'][:250]))])
            except:
                continue
        print('Loading ended.')
    return matrix_of_indicators


def load():
    nasdaq = create_indicators_matrix('NASDAQ', INDICATORS, 10)
    nyse = create_indicators_matrix('NYSE', INDICATORS, 10)
    lse = create_indicators_matrix('LSE', INDICATORS, 10)

    matrix = [*nasdaq, *nyse, *lse]
    labelled_matrix = pd.DataFrame()
    labelled_matrix['stock'] = [stock[0] for stock in matrix]
    labelled_matrix['exchange'] = [stock[1] for stock in matrix]

    rsi = list()
    for stock in range(len(matrix)):
        if matrix[stock][2]['rsi'].values.tolist()[-1] > 69:
            rsi.append(1)
        elif matrix[stock][2]['rsi'].values.tolist()[-1] < 31:
            rsi.append(2)
        else:
            rsi.append(0)
    labelled_matrix['rsi'] = rsi
    labelled_matrix['rsi_value'] = [matrix[stock][2]['rsi'].values.tolist()[-1] for stock in range(len(matrix))]

    bollinger = list()
    for stock in range(len(matrix)):
        # sell if close is near to cross upper bollinger bound
        if matrix[stock][2]['boll_ub'].values.tolist()[-2] > matrix[stock][2]['close'].values.tolist()[-2] and \
                matrix[stock][2]['close'].values.tolist()[-1] >= matrix[stock][2]['boll_ub'].values.tolist()[
            -1] * 0.999:
            bollinger.append(1)
        # buy if close is near to cross lower bollinger bound
        elif matrix[stock][2]['boll_lb'].values.tolist()[-2] < matrix[stock][2]['close'].values.tolist()[-2] and \
                matrix[stock][2]['close'].values.tolist()[-1] <= matrix[stock][2]['boll_lb'].values.tolist()[
            -1] * 1.001:
            bollinger.append(2)
        else:
            bollinger.append(0)
    labelled_matrix['bollinger'] = bollinger
    labelled_matrix['boll_ub_value'] = [matrix[stock][2]['boll_ub'].values.tolist()[-1] for stock in range(len(matrix))]
    labelled_matrix['boll_lb_value'] = [matrix[stock][2]['boll_lb'].values.tolist()[-1] for stock in range(len(matrix))]

    macd = list()
    for stock in range(len(matrix)):
        if 0 > matrix[stock][2]['macd'].values.tolist()[-1] >= matrix[stock][2]['macds'].values.tolist()[-1] * 0.999 and \
                matrix[stock][2]['macd'].values.tolist()[-2] < matrix[stock][2]['macds'].values.tolist()[-2]:
            macd.append(1)
        elif 0 < matrix[stock][2]['macd'].values.tolist()[-1] <= matrix[stock][2]['macds'].values.tolist()[
            -1] * 1.001 and matrix[stock][2]['macd'].values.tolist()[-2] > matrix[stock][2]['macds'].values.tolist()[
            -2]:
            macd.append(2)
        else:
            macd.append(0)
    labelled_matrix['macd'] = macd
    labelled_matrix['macd_value'] = [matrix[stock][2]['macd'].values.tolist()[-1] for stock in range(len(matrix))]

    stochrsi = list()
    for stock in range(len(matrix)):
        if matrix[stock][2]['stochrsi'].values.tolist()[-1] > 80:
            stochrsi.append(1)
        elif matrix[stock][2]['stochrsi'].values.tolist()[-1] < 20:
            stochrsi.append(2)
        else:
            stochrsi.append(0)
    labelled_matrix['stochrsi'] = stochrsi
    labelled_matrix['stochrsi_value'] = [matrix[stock][2]['stochrsi'].values.tolist()[-1] for stock in
                                         range(len(matrix))]

    williams = list()
    for stock in range(len(matrix)):
        if -20 <= matrix[stock][2]['wr'].values.tolist()[-1] <= 0:
            williams.append(1)
        elif -100 <= matrix[stock][2]['wr'].values.tolist()[-1] <= -80:
            williams.append(2)
        else:
            williams.append(0)
    labelled_matrix['williams'] = williams
    labelled_matrix['williams_value'] = [matrix[stock][2]['wr'].values.tolist()[-1] for stock in range(len(matrix))]

    sma = list()
    for stock in range(len(matrix)):
        if matrix[stock][2]['close_50_sma'].values.tolist()[-1] <= matrix[stock][2]['close_200_sma'].values.tolist()[
            -1] * 1.001 and matrix[stock][2]['close_50_sma'].values.tolist()[-2] > \
                matrix[stock][2]['close_200_sma'].values.tolist()[-2]:
            sma.append(1)
        elif matrix[stock][2]['close_50_sma'].values.tolist()[-1] >= matrix[stock][2]['close_200_sma'].values.tolist()[
            -1] * 0.999 and matrix[stock][2]['close_50_sma'].values.tolist()[-2] < \
                matrix[stock][2]['close_200_sma'].values.tolist()[-2]:
            sma.append(2)
        else:
            sma.append(0)
    labelled_matrix['sma'] = sma
    labelled_matrix['sma_50_value'] = [matrix[stock][2]['close_50_sma'].values.tolist()[-1] for stock in
                                       range(len(matrix))]
    labelled_matrix['sma_200_value'] = [matrix[stock][2]['close_200_sma'].values.tolist()[-1] for stock in
                                        range(len(matrix))]

    ao = list()
    for stock in range(len(matrix)):
        if matrix[stock][2]['ao'].values.tolist()[-3] < 0 and matrix[stock][2]['ao'].values.tolist()[-2] < 0 and \
                matrix[stock][2]['ao'].values.tolist()[-1] >= 0:
            ao.append(1)
        elif matrix[stock][2]['ao'].values.tolist()[-3] > 0 and matrix[stock][2]['ao'].values.tolist()[-2] > 0 and \
                matrix[stock][2]['ao'].values.tolist()[-1] <= 0:
            ao.append(2)
        else:
            ao.append(0)
    labelled_matrix['ao'] = ao
    labelled_matrix['ao_value'] = [matrix[stock][2]['ao'].values.tolist()[-1] for stock in range(len(matrix))]

    adx = list()
    for stock in range(len(matrix)):
        if matrix[stock][2]['dx'].values.tolist()[-1] >= 25 and matrix[stock][2]['pdi'].values.tolist()[-1] < \
                matrix[stock][2]['ndi'].values.tolist()[-1] * 1.001 and matrix[stock][2]['ndi'].values.tolist()[-2] < \
                matrix[stock][2]['pdi'].values.tolist()[-2]:
            adx.append(1)
        elif matrix[stock][2]['dx'].values.tolist()[-1] >= 25 and matrix[stock][2]['pdi'].values.tolist()[-1] > \
                matrix[stock][2]['ndi'].values.tolist()[-1] * 0.999 and matrix[stock][2]['ndi'].values.tolist()[-2] > \
                matrix[stock][2]['pdi'].values.tolist()[-2]:
            adx.append(2)
        else:
            adx.append(0)
    labelled_matrix['adx'] = adx

    labelled_matrix['close'] = [matrix[stock][2]['close'].values.tolist()[-1] for stock in range(len(matrix))]
    labelled_matrix['open'] = [matrix[stock][2]['open'].values.tolist()[-1] for stock in range(len(matrix))]
    labelled_matrix['high'] = [matrix[stock][2]['high'].values.tolist()[-1] for stock in range(len(matrix))]
    labelled_matrix['low'] = [matrix[stock][2]['low'].values.tolist()[-1] for stock in range(len(matrix))]
    labelled_matrix['volume'] = [matrix[stock][2]['volume'].values.tolist()[-1] for stock in range(len(matrix))]

    for column in range(50):
        labelled_matrix['close_' + str(column)] = [matrix[stock][2]['close'].values.tolist()[-(50 - column)] for stock
                                                   in range(len(matrix))]
    scaler_c = Scaler()
    model_loaded = TiDEModel.load("../ml_stocks_classifier/model_TiDEModel_ema.pt",
                                  map_location="cpu")
    model_loaded.to_cpu()

    labelled_matrix['close_scaled'] = [
        scaler_c.fit_transform(TimeSeries.from_values(matrix[stock][2]['close'].values[:250])) for stock in
        range(len(matrix))]

    buy_icon = '🟩'
    sell_icon = '🟥'
    neutral = '⬜️'

    predictions = []
    for stock in range(len(labelled_matrix)):
        prediction = model_loaded.predict(n=5, series=labelled_matrix['close_scaled'][stock])
        prices = [np.array(labelled_matrix['close_scaled'].values[0].data_array()).flatten()[-1],
                  *np.array(prediction.data_array()).flatten()]
        moves = list()
        rises = 0
        for day in range(len(prices) - 1):
            if prices[day + 1] < prices[day]:
                moves.append(sell_icon)
            elif prices[day + 1] > prices[day]:
                moves.append(buy_icon)
                rises += 1
            else:
                moves.append(neutral)
        predictions.append([*moves, rises])

    for column in range(5):
        labelled_matrix['prediction_' + str(column)] = [predictions[stock][column] for stock in range(len(predictions))]
    labelled_matrix['rises'] = [predictions[stock][5] for stock in range(len(predictions))]
    return labelled_matrix
