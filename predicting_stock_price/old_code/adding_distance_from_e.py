# %%
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
from data_generation import get_data
from feature_engineering import featureEng

# %%


NAME = 'MVNE'
random_state_plot = 3
test_size = 0.15
years_back = 1.4
threshold_days_in_week = 1
min_prcnt_data = 0.8
min_days_in_week = 1
corr_threshold = 0.35
corr_inter_threshold = 0.7
days_interval = 6
date_reference = '2020-08-01'
models_path = './pickel_models'
corr_inter_threshold_main = corr_threshold

# %%
main_ticker = 'MVNE'
dfr = get_data(ticker_name=NAME, data_from_csv=1, path_from='raw_data')
dfc = get_data(ticker_name=NAME, data_from_csv=1, path_from='data')

d = featureEng(main_ticker,
               years_back=years_back,
               data_from_csv=True,
               path_from='data')

# %%
df = d.daily_diff()
print('\n-- df.columns --\n', df.columns, '\n')
print('-- df.shape --\n', df.shape, '\n-------')
print(df.head(2))

# %%
df_weekly = d.weekly_mean(ticker_name=main_ticker,
                          df=df,
                          start_date=date_reference,
                          days_interval=6,
                          threshold_days_in_week=2,
                          #   roll_num=3
                          )

# %%
df_weekly

expected_value_rang = 7
dfc_close = dfc[['close']]

dfc_close['roll'] = dfc_close.rolling(f'{expected_value_rang}d').mean()

dfc_close['diff'] = dfc_close['close'] - dfc_close['roll']
dfc_close['diff'] = dfc_close['diff'].rolling(f'3d').mean()

# %%
import matplotlib.pyplot as plt

dfc_close.plot(style='-o')
plt.show()
# %%
dff = df_reg[:]
inx = dff.index
# %%
import pandas as pd

max_date_in_rang = []
date_ranges = []
for dd in inx:
    d = [pd.Timestamp(j) for j in dd]
    max_date_in_rang.append(max([i for i in dfc_close.index if ((i >= d[0]) and (i <= d[1]))]))
    date_ranges.append(dd)

# %%
df_test = dfc_close[:]
df_test = df_test.loc[max_date_in_rang, :'diff']
df_test.index = date_ranges
df_test
# %%
dff['diff'] = df_test['diff']

df_f = dff[['diff', f'{main_ticker}_main_symb_weekly_mean_diff']]
# %%
df_p = df_f[:]
df_p.index = [f'{i[0]}\n{i[1]}' for i in df_p.index]
df_p.plot(x='diff', y='MVNE_main_symb_weekly_mean_diff',style='o')
plt.show()

# %%
df_p.corr()