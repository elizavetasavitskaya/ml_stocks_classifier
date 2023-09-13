import pandas as pd
import numpy as np
import json

from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from tqdm.notebook import tqdm
import datetime

import logging

logging.disable(logging.CRITICAL)

import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel, NLinearModel, TransformerModel
from pytorch_lightning.callbacks import Callback, EarlyStopping
from darts.metrics import mape, smape, mae

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

torch.manual_seed(1);


def get_json_parsed_data(url):
    """Return json object from https://financialmodelingprep.com/ url"""
    return json.loads(urlopen(url).read().decode('utf-8'))


def load_stocks_list(exchange):
    stocks = pd.DataFrame(get_json_parsed_data(traded_stocks__url))
    stocks = stocks[(stocks.exchangeShortName == exchange) & (stocks.type == 'stock')].symbol.tolist()

def create_prices_dataset(exchange, number_of_treads):
    """Download last year prices (~250 days) for specific exchange,
  exchanges are processed by separate models because of lots common
  features between specific exchange stocks prices behavior
  Loading for NASDAQ tackes around 7 mins on colab"""
    traded_stocks__url = 'https://financialmodelingprep.com/api/v3/available-traded/list?apikey=1401249d9cdf8f975028aa69f2ef15c7'
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365)

    def load_close_prices_and_volumes(stock):
        try:
            price_url = ('https://financialmodelingprep.com/api/v4/historical-price-adjusted'
                         f'/{stock}/1/day/{start}/{end}?apikey=1401249d9cdf8f975028aa69f2ef15c7')
            data = pd.DataFrame(get_json_parsed_data(price_url)['results'])
            close, volume = data[['c']], data[['v']]
            close.fillna(method='ffill', inplace=True)
            volume.fillna(method='ffill', inplace=True)
            if len(close) >= 250 and close['c'][0] >= 5 and volume['v'][0] >= 20000:
                close = np.flip(close['c'].values[:250])
                volume = np.flip(volume['v'].values[:250])
                return [stock, {'close': close, 'volume': volume}]
        except KeyError or IndexError or TypeError:
            return None
        return None

    stocks_prices = {}
    stocks = load_stocks_list(exchange)
    with ThreadPoolExecutor(max_workers=number_of_treads) as executor:
        for data in tqdm(executor.map(load_close_prices_and_volumes, stocks),
                         desc=f'load_close_prices_and_volumes {exchange}', total=len(stocks)):
            if data:
                scaler_c, scaler_v = Scaler(), Scaler()
                stocks_prices[data[0]] = data[1]
                stocks_prices[data[0]]['scaled_close'] = scaler_c.fit_transform(
                    TimeSeries.from_values(data[1]['close']));
                stocks_prices[data[0]]['scaled_volume'] = scaler_v.fit_transform(
                    TimeSeries.from_values(data[1]['volume']));
    return stocks_prices


def train_TSF_model_and_predict():
    prices_dataset_nasdaq = create_prices_dataset('NASDAQ', 2)
    stocks_nasdaq = list(prices_dataset_nasdaq.keys())[:100]
    train_close_nasdaq = list()
    for stock in stocks_nasdaq:
        train_close_nasdaq.append(prices_dataset_nasdaq[stock]['scaled_close'])

    early_stopper = EarlyStopping("train_loss", min_delta=0.0001, patience=5, verbose=True)

    model_NHits = NHiTSModel(force_reset=True, input_chunk_length=5, output_chunk_length=1,
                       n_epochs=20, random_state=0, optimizer_kwargs={"lr": 1e-3}, model_name="NLinearModel",
                       pl_trainer_kwargs={"callbacks": [early_stopper]}, save_checkpoints=True)
    history = model_NHits.fit(series=train_close_nasdaq, verbose=True);
    model_NHits.save('price_nasdaq_model.pt')


train_TSF_model_and_predict()
