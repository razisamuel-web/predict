# %%
from datetime import date, timedelta, datetime
from feature_engineering import featureEng
from data_generation import get_data
from train import Train, ClfTrain
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
import pickle
from predict import Predict
from general_functions import next_week_full_train_behavior, weekly_correlations_to_csv
import matplotlib.pyplot as plt
import copy
import time
from general_functions import weekly_correlations_to_csv
import json
import seaborn as sns

sns.set_style("darkgrid")

# Read configs
configs = json.loads(open('configs.json', 'r').read())
for k, v in configs.items():
    exec(f"{k} = '{v}'") if type(v) == str else exec(f"{k} = {v}")

NAME = configs["NAME"]
random_state_plot = configs["random_state_plot"]
test_size = configs["test_size"]
years_back = configs["years_back"]
threshold_days_in_week = configs["threshold_days_in_week"]
min_percentage_data = configs["min_percentage_data"]
min_days_in_week = configs["min_days_in_week"]
corr_threshold = configs["corr_threshold"]
corr_inter_threshold = configs["corr_inter_threshold"]
days_interval = configs["days_interval"]
models_path = configs["models_path"]
corr_inter_threshold_main = configs["corr_inter_threshold_main"]
date_reference = configs["date_reference"]
date_reference_end = configs["date_reference_end"]
correlation_path = "./weekly_diff_test"

# Read ticker list
df_tickers = pd.read_csv('./symbols/israeli_symbols_names.csv')
tickers = list(df_tickers['Symbol'])

with open("./symbols/clean_ticker_list.txt", "rb") as fp:
    clean_ticker_list = pickle.load(fp)
    print(f'number of ticker in the beegining {len(tickers)}')
    print(f'number of ticker after validation {len(clean_ticker_list)}')
    print(f'diff is = {len(tickers) - len(clean_ticker_list)}')

measurements_l = ['date_reference',
                  'next_week_price_full_train',
                  'predicted_stock_class',
                  'r2_test',
                  'r2_train',
                  'r2_train_full_train',
                  'rmse_train_full_train',
                  'False_p',
                  'True_p',
                  'predicted_diff_full_train',
                  'percentage_change_full_train'
                  ]

measurements = {i: [] for i in measurements_l}
measurements

dfr = get_data(ticker_name=NAME, data_from_csv=1, path_from='raw_data')
dfc = get_data(ticker_name=NAME, data_from_csv=1, path_from='data')

df_weekly = get_data(NAME,
                     data_from_csv=True,
                     path_from='weekly_diff',
                     set_as_index=['first_day_in_week', 'last_day_in_week'],
                     index_type='object')

# %%
while date_reference < date_reference_end:
    print(date_reference)

    d = featureEng(NAME,
                   date_reference=date_reference,
                   years_back=years_back,
                   data_from_csv=True,
                   path_from='data')

    # df = d.daily_diff()

    # df = d.weekly_mean(ticker_name=NAME,
    #                    df=df,
    #                    start_date=date_reference,
    #                    days_interval=days_interval,
    #                    threshold_days_in_week=threshold_days_in_week
    #                    )

    print(f'df_weekly.shpe{df_weekly.shape}\ndf_weekly.index[-1]{df_weekly.index[-1]}\n'
          f'df_weekly.index[0]{df_weekly.index[0]}')
    # Retrieve rows from given time period - (Cutting upper and lower tails)

    same_date_last_year = str(datetime.strptime(date_reference, "%Y-%m-%d") - timedelta(days=round(years_back * 365)))

    dates_first = df_weekly.reset_index()['first_day_in_week']
    dates_last = df_weekly.reset_index()['last_day_in_week']

    lower = dates_first[dates_first >= same_date_last_year].index[0]
    upper = dates_last[dates_last <= date_reference].index[-1] + 1  # since its started from 0
    df_weekly = df_weekly.iloc[lower:upper, :]

    with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
        clean_ticker_list = pickle.load(fp)

    new_list = weekly_correlations_to_csv(tickers_list=clean_ticker_list,
                                          years_back_data_generation=years_back,
                                          start_date=date_reference,
                                          days_interval=days_interval,
                                          threshold_days_in_week=threshold_days_in_week,
                                          path='./weekly_diff')

    with open("new_list_rolled_tickes.txt", "w") as file:
        file.write(str(new_list))

    with open("new_list_rolled_tickes.txt", "r") as file:
        clean_ticker_list = eval(file.readline())

    df_corr, low_week_sampels_dict = d.weekly_correlation(df_weekly_mean=df_weekly,
                                                          tickers_list=clean_ticker_list,
                                                          date_reference=date_reference,
                                                          min_prcnt_data=min_percentage_data,
                                                          threshold=min_days_in_week,
                                                          path_from='weekly_diff',
                                                          set_as_index=['first_day_in_week', 'last_day_in_week']
                                                          )

    start_time = time.time()
    df_reg_full = d.reg_df(
        ticker_name=NAME,
        df_weekly=df_weekly,
        df_corr=df_corr,
        start_date=date_reference,
        threshold=corr_threshold,
        # activate_automated_rolling=True
    )

    print("\n--- %s seconds ---" % (time.time() - start_time), 'df_reg_full.shape = ', df_reg_full.shape)

    start_time = time.time()

    try:
        df_reg_full = d.df_reg_int(df_reg=df_reg_full,
                                   corr_inter_threshold=corr_inter_threshold,
                                   corr_inter_threshold_main=corr_inter_threshold_main)

        df_reg = copy.copy(df_reg_full[:-1])
        inter_columns = [inter for inter in df_reg.columns if 'INT' in inter]
        number_of_inter = len(inter_columns)

        start = time.time()

        train = Train(NAME,
                      df_reg=df_reg,
                      test_size=test_size,
                      path=models_path)

        train_dict = train.df_filtered_dict

        dict_reg_results = {'r2_test': train_dict['r2_test'],
                            'r2_train': train_dict['r2_train'],
                            'alpha': train_dict['alpha'],
                            'rmse_train': train_dict['rmse_train'],
                            'corra_mean': train_dict['corra_mean'],
                            'predictor_num': train_dict['predictor_num']
                            }

        # class
        clftrain = ClfTrain(tick_name=NAME)
        clf = clftrain.fit_lr_gridsearch_cv()
        summary_dict = clftrain.generate_clf_summary(clf, classifire_type='lr')

        df_pred_actual = clftrain.predict_actual_diffs(clf)

        reg = train.reg
        colsl = train_dict['current_corrs_str']
        df_reg_full = df_reg_full[colsl]
        target_column_name = f'{NAME}_main_symb_weekly_mean_diff'

        predict = Predict(reg,
                          target_column_name,
                          df_reg=df_reg_full,
                          date_reference=date_reference,
                          cols=colsl,
                          days_interval=days_interval)

        next_week_behavior = predict.next_week_behavior(df=d._df,
                                                        date_reference=date_reference)

        next_week_class = predict.next_week_class(clf)

        r = next_week_full_train_behavior(main_ticker=NAME,
                                          df_reg_full=df_reg_full,
                                          df_raw=d._df,
                                          cols=colsl,
                                          train_object=train,
                                          clftrain_object=clftrain,
                                          days_interval=days_interval,
                                          date_reference=date_reference
                                          )

        df_reg = df_reg[colsl]

        # Results from 'train_dict' = partial train object
        r2_test = train_dict['r2_test']
        r2_train = train_dict['r2_train']

        # Results from 'r' = full train object
        r2_train_full_train = r['r2_train_full_train']
        predicted_diff_full_train = r['predicted_diff_full_train']
        percentage_change_full_train = r['percentage_change_full_train']
        rmse_train_full_train = r['rmse_train_full_train']
        next_week_price_full_train = r['next_week_price_full_train']
        predicted_stock_class = r['class']
        False_p = r['False_p']
        True_p = r['True_p']

        for k in measurements.keys():
            measurements[k].append(eval(k))

    except:
        if df_reg_full.shape[1] == 2:
            for k in measurements.keys():
                measurements[k].append(None) if k != 'date_reference' else measurements[k].append(date_reference)
        else:
            print('df_reg_full columns number is bigger the 1 something else happened')
            break

    date_reference = str(datetime.strptime(date_reference, "%Y-%m-%d").date() + timedelta(days=days_interval + 1))

    print(measurements)

df_measurements = pd.DataFrame(measurements)
df_measurements.to_csv('df_measurements_nons.csv')

# %%
df_measurements = pd.read_csv('df_measurements.csv').drop('Unnamed: 0', axis=1)
# %%

# Adding: actual prices prioed after, actual date period after, and std by adding and deleting n periods back
n = 10
df_measurements = df_measurements.set_index('date_reference')
dict = {'actual_date': [], 'actual_close_price': [], 'measurments_df_index': []}
# %%

dates = [str(datetime.strptime(df_measurements.index[0], "%Y-%m-%d").date() - timedelta(days=(days_interval + 1) * i))
         for i in
         range(1, n)][::-1] + list(df_measurements.index)
# %%
list(df_measurements.index)
# %%
for i, j in zip(dates[:-1], dates[1:]):
    j = str(datetime.strptime(j, "%Y-%m-%d").date() - timedelta(days=1))

    max_exist_date = max(dfc.index[dfc.index <= datetime.strptime(j, "%Y-%m-%d")])
    dict['actual_date'].append(max_exist_date)
    dict['actual_close_price'].append(round(dfc.loc[max_exist_date, 'close'], 2))
    dict['measurments_df_index'].append(i)

df_stat = pd.DataFrame(dict).set_index('measurments_df_index')
df_stat['std'] = df_stat.actual_close_price.rolling(n).std()
df_stat = df_stat.iloc[n - 1:, :]
df_measurements = pd.concat([df_measurements, df_stat], axis=1)

# multiply number of days in week in predicted average diff
df_measurements_index = df_measurements.actual_date.astype(str)
for i in df_measurements_index[:-1]:
    f = df_weekly.index.get_level_values('first_day_in_week') >= i
    l = df_weekly.index.get_level_values('last_day_in_week') <= i
    if any(f == l):
        if i in df_measurements_index.to_list():
            df_measurements.loc[df_measurements_index == i, 'days_in_week'] = df_weekly[f == l].days_in_week[0]

# metrices of predicted week : std, number of days in week
df_measurements.loc[df_measurements.index[-1], 'days_in_week'] = 5
df_measurements.loc[df_measurements.index[-1], 'std'] = df_stat['std'][-1]

df_measurements[
    'next_week_price_full_train_mult'] = df_measurements.next_week_price_full_train + df_measurements.predicted_diff_full_train * (
        df_measurements.days_in_week - 1)

df_measurements['predicted_diff_full_train_mult'] = df_measurements.predicted_diff_full_train * (
        df_measurements.days_in_week - 1)

# Adding actual diff
df_measurements['actual_diff'] = df_measurements['actual_close_price'].diff(1)

# Adding upper & lower CI bounds
df_measurements['upper'] = df_measurements.next_week_price_full_train_mult + 1.645 * (
            df_measurements['std'] / np.sqrt(n))
df_measurements['lower'] = df_measurements.next_week_price_full_train_mult - 1.645 * (
            df_measurements['std'] / np.sqrt(n))

# %% Plot

# df_pred_vs_actual = df_pred_vs_actual.set_index(['actual_date'], drop=True)
columns_to_plot1 = ['actual_close_price',
                    'predicted_diff_full_train_mult',
                    'actual_diff']
d = df_measurements[columns_to_plot1]
ax = d.plot(style='-o', color=['C1', 'C3', 'C4'])

columns_to_plot2 = 'next_week_price_full_train_mult'
df_predicted = df_measurements[columns_to_plot2]
predicted_color, predicted_alpha = 'C0', .25
ax = df_predicted.plot(ax=ax, style='-o', color=[predicted_color], alpha=predicted_alpha)
plt.fill_between(x=df_measurements.index,
                 y1=df_measurements['upper'],
                 y2=df_measurements['lower'],
                 color='C0',
                 alpha=predicted_alpha)

ax.legend()
plt.show()

# %% TODO adding try except in case that there is no df_reg for model and think which prediction we want to use in this date and if we want to nark this predections

fig = ax.get_figure()
fig.savefig('matplotlmatib_figure.png')   # save the figure to file
plt.close(fig)