# %%
from feature_engineering import featureEng
from data_generation import get_data
from train import Train, ClfTrain
import pandas as pd
import pickle
from predict import Predict
from general_functions import corrs_mean_from_cols_names, next_week_full_train_behavior, weekly_correlations_to_csv
from plots import plot_pred_actual
import copy
import time

NAME = 'LSCO'
random_state_plot = 3
test_size = 0.15
years_back = 1
threshold_days_in_week = 1
min_prcnt_data = 0.8
min_days_in_week = 1
corr_threshold = 0.30
corr_inter_threshold = 0.7
corr_inter_threshold_main = corr_threshold
date_reference = '2020-07-11'
models_path = './pickel_models'

df_tickers = pd.read_csv('./symbols/israeli_symbols_names.csv')
tickers = list(df_tickers['Symbol'])

with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
    clean_ticker_list = pickle.load(fp)
    print(f'number of ticker in the beegining {len(tickers)}')
    print(f'number of ticker after validation {len(clean_ticker_list)}')
    print(f'diff is = {len(tickers) - len(clean_ticker_list)}')

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

df_weekly = get_data(main_ticker,
                     data_from_csv=True,
                     path_from='weekly_diff',
                     set_as_index='week_of_year',
                     index_type='object')

print('\n-- df_weekly.columns --\n', df_weekly.columns, '\n')
print('-- df_weekly.shape--\n', df_weekly.shape, '\n-------')
df_weekly.head(2)

df_corr, low_week_sampels_dict = d.weekly_correlation(df_weekly_mean=df_weekly,
                                                      tickers_list=clean_ticker_list,
                                                      min_prcnt_data=min_prcnt_data,
                                                      threshold=min_days_in_week)  # <---- low_week_sampels_dict

print('\n-- df_corr.columns--\n', df_corr.columns, '\n')
print('-- df_corr.shape --\n', df_corr.shape, '\n\n-------')

df_reg_full = d.reg_df(df_weekly,
                       df_corr,
                       threshold=corr_threshold)

start_time = time.time()
df_reg_full = d.df_reg_int(df_reg=df_reg_full,
                           corr_inter_threshold=corr_inter_threshold,
                           corr_inter_threshold_main=corr_inter_threshold_main)

print("--- %s seconds ---" % (time.time() - start_time))

df_reg = copy.copy(df_reg_full[:-1])

df_reg.drop(['first_day_in_week', 'last_day_in_week'], axis=1, inplace=True)
cols = df_reg.columns

inter_columns = [inter for inter in df_reg.columns if 'int' in inter]
number_of_inter = len(inter_columns)

print(f"inter columns {inter_columns}")
print(f'\n----- number of inter actions {number_of_inter} ----')

print('\n-- df_reg.columns --\n', df_reg.columns, '\n')
print('-- df_reg.shape--\n', df_reg.shape, '\n-------')
print(df_reg.head(2))
print(cols)

corrs, corrs_mean = corrs_mean_from_cols_names(cols)
corrs_mean

target_column_name = f'{main_ticker}_main_symb_weekly_mean_diff'

#%%
df_reg.shape
# %%
start = time.time()
train = Train(NAME,
              df_reg=df_reg,
              test_size=test_size,
              path=models_path)

train_dict = train.df_filtered_dict
pkl_filename = train_dict['reg']

print('====  r2_train  ====\n')
print(train_dict['r2_train'], '\n')
print('====  r2_test  ====\n')
print(train_dict['r2_test'], '\n')
print('====  alpha  ====\n')
print(train_dict['alpha'], '\n')
print('====  rmse_train  ====\n')
print(train_dict['rmse_train'], '\n')
print('\n====  corra_mean  ====\n')
print(train_dict['corra_mean'])
print('\n====  predictor_num  ==== \n')
print(train_dict['predictor_num'], '\n')

with open(pkl_filename, 'rb') as file:
    reg = pickle.load(file)

colsl = train_dict['current_corrs_str']
df_reg = df_reg[colsl]

df_reg_full = df_reg_full[colsl + ['first_day_in_week', 'last_day_in_week']]

print('time in seconds ', round(time.time() - start, 1))
# %%

len(reg.coef_), df_reg.shape, len(colsl)

# %% classifier\
clftrain = ClfTrain(tick_name=main_ticker)
clf = clftrain.fit_lr_gridsearch_cv()

# %%
summary_dict = clftrain.generate_clf_summary(clf, classifire_type='lr')
print(summary_dict['classification_report'], '\n\n', f"score : {summary_dict['accuracy_test']}")

# %%
df_pred_actual = clftrain.predict_actual_diffs(clf)
print(df_pred_actual)

# %%
df_corrs_summary = train.df_corrs_summary

predict = Predict(reg,
                  target_column_name,
                  df_reg=df_reg_full,
                  date_reference=date_reference,
                  cols=colsl)  # <--datetime.today()

x_last = predict._x_sample_row
x_last.columns

next_week_behavior = predict.next_week_behavior(df=d._df,
                                                date_reference=date_reference)

print('\n\n--   next_week_behavior   --\n', pd.DataFrame.from_dict(next_week_behavior, orient='index'))
# %%
print('\n\n--   next week class   --')
next_week_class = predict.next_week_class(clf)
pd.DataFrame.from_dict(next_week_class, orient='index').T

# %%
alpha = train_dict['alpha']
r = next_week_full_train_behavior(main_ticker=main_ticker,
                                  df_reg_full=df_reg_full,
                                  alpha=alpha,
                                  df_row=d._df,
                                  cols=colsl,
                                  date_reference=date_reference)

print('\n\n--  next_week_full_train_behavior_r   --\n', pd.DataFrame.from_dict(r, orient='index'))

plot_pred_actual(df_regression=df_reg,
                 main_ticker=main_ticker,
                 reg=reg,
                 test_size=test_size,
                 random_state=random_state_plot)
