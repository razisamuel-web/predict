# %%
from datetime import date, timedelta, datetime
from feature_engineering import featureEng
from data_generation import get_data
from train import Train, ClfTrain
import pandas as pd
import pickle
from predict import Predict
from general_functions import next_week_full_train_behavior, weekly_correlations_to_csv
from plots import plot_pred_actual
import copy
import time


NAME = 'EDRL'
random_state_plot = 3
test_size = 0.15
years_back = 1.0
threshold_days_in_week = 1
min_prcnt_data = 0.8
min_days_in_week = 1
corr_threshold = 0.35
corr_inter_threshold = 0.7
days_interval = 6
date_reference = '2020-08-08'
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
               date_reference=date_reference,
               years_back=years_back,
               data_from_csv=True,
               path_from='data')

# %%
df = d.daily_diff()
print('\n-- df.columns --\n', df.columns, '\n')
print('-- df.shape --\n', df.shape, '\n-------')
print(df.head(2))

# %%
df = d.weekly_mean(ticker_name=main_ticker,
                   df=df,
                   start_date=date_reference,
                   days_interval=days_interval,
                   threshold_days_in_week=threshold_days_in_week
                   )

# %%
pd.set_option('display.max_rows', 500)
df_weekly = get_data(main_ticker,
                     data_from_csv=True,
                     path_from='weekly_diff',
                     set_as_index=['first_day_in_week', 'last_day_in_week'],
                     index_type='object')

# Retrieve rows from given time period - (Cutting upper and lower tails)

same_date_last_year = str(datetime.strptime(date_reference, "%Y-%m-%d") - timedelta(days=round(years_back * 365)))
same_date_last_year, date_reference

dates_first = df_weekly.reset_index()['first_day_in_week']
dates_last = df_weekly.reset_index()['last_day_in_week']

lower = dates_first[dates_first >= same_date_last_year].index[0]
upper = dates_last[dates_last <= date_reference].index[-1] + 1  # since its started from 0
df_weekly = df_weekly.iloc[lower:upper, :]

# TODO - The aggregation of all the df_weekly are shoulde be with respect to the date reference ,
#  for example we cant just cut df which started in 8/09 from df which started in 12/09, from that reasone we should run all the process on all the stocks each time


# %%
with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
    clean_ticker_list = pickle.load(fp)



# %%
from general_functions import weekly_correlations_to_csv

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


# %%
df_corr, low_week_sampels_dict = d.weekly_correlation(df_weekly_mean=df_weekly,
                                                      tickers_list=clean_ticker_list,
                                                      date_reference=date_reference,
                                                      min_prcnt_data=min_prcnt_data,
                                                      threshold=min_days_in_week,
                                                      path_from='weekly_diff',
                                                      set_as_index=['first_day_in_week', 'last_day_in_week']
                                                      )

print('\n-- df_corr.columns--\n', df_corr.columns, '\n')
print('-- df_corr.shape --\n', df_corr.shape, '\n\n-------')

# %%
start_time = time.time()
df_reg_full = d.reg_df(
    ticker_name=main_ticker,
    df_weekly=df_weekly,
    df_corr=df_corr,
    start_date=date_reference,
    threshold=corr_threshold,
    # activate_automated_rolling=True
)

print("\n--- %s seconds ---" % (time.time() - start_time), 'df_reg_full.shape = ', df_reg_full.shape)

# %%
start_time = time.time()
# the current inter function remove same ticks inter to prevent from multicollinearity
df_reg_full = d.df_reg_int(df_reg=df_reg_full,
                           corr_inter_threshold=corr_inter_threshold,
                           corr_inter_threshold_main=corr_inter_threshold_main)

print("--- %s seconds ---" % (time.time() - start_time))

# %%
df_reg = copy.copy(df_reg_full[:-1])
inter_columns = [inter for inter in df_reg.columns if 'INT' in inter]
number_of_inter = len(inter_columns)

print(f"inter columns {inter_columns[1:10]}")
print(f'\n----- number of interactions : == >>  {number_of_inter} ----\n')

print('\n-- df_reg.columns --\n\n', df_reg.columns, '\n')
print('-- df_reg.shape--\n', df_reg.shape, '\n-------')

# %%
start = time.time()

train = Train(NAME,
              df_reg=df_reg,
              test_size=test_size,
              path=models_path)

train_dict = train.df_filtered_dict

print('time in seconds ', round(time.time() - start, 1))

dict_reg_results = {'r2_test': train_dict['r2_test'],
                    'r2_train': train_dict['r2_train'],
                    'alpha': train_dict['alpha'],
                    'rmse_train': train_dict['rmse_train'],
                    'corra_mean': train_dict['corra_mean'],
                    'predictor_num': train_dict['predictor_num']
                    }

print(pd.DataFrame.from_dict(dict_reg_results, orient='index'))

# %% class
clftrain = ClfTrain(tick_name=main_ticker)
clf = clftrain.fit_lr_gridsearch_cv()
summary_dict = clftrain.generate_clf_summary(clf, classifire_type='lr')
print(summary_dict['classification_report'], '\n\n', f"score : {summary_dict['accuracy_test']}")

df_pred_actual = clftrain.predict_actual_diffs(clf)
print(df_pred_actual)

# %%
reg = train.reg
colsl = train_dict['current_corrs_str']
df_reg_full = df_reg_full[colsl]
target_column_name = f'{main_ticker}_main_symb_weekly_mean_diff'

predict = Predict(reg,
                  target_column_name,
                  df_reg=df_reg_full,
                  date_reference=date_reference,
                  cols=colsl,
                  days_interval=days_interval)

next_week_behavior = predict.next_week_behavior(df=d._df,
                                                date_reference=date_reference)

print('\n\n--   next_week_behavior   --\n', pd.DataFrame.from_dict(next_week_behavior, orient='index'))

next_week_class = predict.next_week_class(clf)
print('\n\n--   next week class   --\n', pd.DataFrame.from_dict(next_week_class, orient='index'))

# %%
r = next_week_full_train_behavior(main_ticker=main_ticker,
                                  df_reg_full=df_reg_full,
                                  df_raw=d._df,
                                  cols=colsl,
                                  train_object=train,
                                  clftrain_object=clftrain,
                                  days_interval=days_interval,
                                  date_reference=date_reference
                                  )

print('\n\n--  next_week_full_train_behavior_r   --\n', pd.DataFrame.from_dict(r, orient='index'))

# %%
df_reg = df_reg[colsl]
plot_pred_actual(df_regression=df_reg,  # TODO TAKE INDEX ON WEEK FOREWORD
                 main_ticker=main_ticker,
                 test_size=test_size,
                 random_state=random_state_plot,
                 reg=train.reg)
