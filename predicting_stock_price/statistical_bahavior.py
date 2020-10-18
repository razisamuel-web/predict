# %%
from feature_engineering import featureEng
from data_generation import get_data
from train import Train, ClfTrain
import pandas as pd
import pickle
from datetime import datetime, timedelta
from predict import Predict
from general_functions import next_week_full_train_behavior, weekly_correlations_to_csv
from plots import plot_pred_actual
import copy
import time

ds
NAME = 'ECP'
random_state_plot = 3
test_size = 0.15
years_back = 1.4
threshold_days_in_week = 1
min_prcnt_data = 0.8
min_days_in_week = 1
corr_threshold = 0.3
corr_inter_threshold = 0.7
days_interval = 6
date_reference = '2020-07-25'
models_path = './pickel_models'
corr_inter_threshold_main = corr_threshold

df_tickers = pd.read_csv('./symbols/israeli_symbols_names.csv')
tickers = list(df_tickers['Symbol'])

with open("./symbols/clean_ticker_list.txt", "rb") as fp:
    clean_ticker_list = pickle.load(fp)
    print(f'number of ticker in the beegining {len(tickers)}')
    print(f'number of ticker after validation {len(clean_ticker_list)}')
    print(f'diff is = {len(tickers) - len(clean_ticker_list)}')

# %%
main_ticker = NAME
dfr = get_data(ticker_name=NAME, data_from_csv=1, path_from='raw_data')
dfc = get_data(ticker_name=NAME, data_from_csv=1, path_from='data')

d = featureEng(main_ticker,
               years_back=years_back,
               data_from_csv=True,
               path_from='data')

df = d.daily_diff()
print('\n-- df.columns --\n', df.columns, '\n')
print('-- df.shape --\n', df.shape, '\n-------')
print(df.head(2))

# %%
df.columns
# %%

df['rolling'] = df['diff_ECP'].rolling(2, min_periods=1).std()

# %%
import numpy as np

np.std(df['diff_ECP'][0:2])

# %%
df['diff_ECP'][0:2]
# %%
df['diff_ECP'][0:2].std(ddof=0)
# %%
df.shape, dfc.shape
# %%
df_rolled = dfc[['close']]
df_rolled['mean_close'] = dfc[['close']].rolling(3, min_periods=1).mean()
df_rolled['std_close'] = dfc[['close']].rolling(10, min_periods=1).std(ddof=0)

df_rolled[f'diff_{NAME}'] = df[f'diff_{NAME}']
df_rolled[f'mean_diff_{NAME}'] = df[f'diff_{NAME}'].rolling(3, min_periods=1).mean()
df_rolled[f'std_diff_{NAME}'] = df[f'diff_{NAME}'].rolling(10, min_periods=1).std(ddof=0)

df_rolled
# %%

