# %%

from feature_engineering import featureEng
from data_generation import get_data
from old_code.dev_train import Trainn
from train import Train
import pandas as pd
import pickle
import copy
import time

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
    tickers = pickle.load(fp)

dd = {'tick': [],

      'rmse_train_c': [],
      'rmse_train_p': [],
      'rmse_train_diff': [],

      'rmse_test_c': [],
      'rmse_test_p': [],
      'rmse_test_diff': [],

      'r2_test_c': [],
      'r2_test_p': [],
      'r2_test_diff': [],

      'r2_train_c': [],
      'r2_train_p': [],
      'r2_train_diff': [],

      'pred_num_c': [],
      'pred_num_p': [],
      'pred_num_diff': [], }

for j, n in enumerate(tickers):
    try:
        print(len(tickers) - j, '    =====   ', n)
        NAME = n
        random_state_plot = 3
        test_size = 0.15
        years_back = 1.4
        threshold_days_in_week = 1
        min_prcnt_data = 0.8
        min_days_in_week = 1
        corr_threshold = 0.32
        corr_inter_threshold = 0.7
        days_interval = 6
        date_reference = '2020-08-01'
        models_path = './pickel_models'
        corr_inter_threshold_main = corr_threshold

        df_tickers = pd.read_csv('./symbols/israeli_symbols_names.csv')
        tickers = list(df_tickers['Symbol'])

        with open("./symbols/clean_ticker_list.txt", "rb") as fp:
            clean_ticker_list = pickle.load(fp)
        main_ticker = NAME
        dfr = get_data(ticker_name=NAME, data_from_csv=1, path_from='raw_data')
        dfc = get_data(ticker_name=NAME, data_from_csv=1, path_from='data')

        d = featureEng(main_ticker,
                       years_back=years_back,
                       data_from_csv=True,
                       path_from='data')

        df = d.daily_diff()

        df, df_roll = d.weekly_mean(ticker_name=main_ticker,
                                    df=df,
                                    start_date=date_reference,
                                    days_interval=6,
                                    threshold_days_in_week=2,
                                    #   roll_num=3
                                    )

        df_weekly = get_data(main_ticker,
                             data_from_csv=True,
                             path_from='weekly_diff',
                             set_as_index=['first_day_in_week', 'last_day_in_week'],
                             index_type='object')

        with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
            clean_ticker_list = pickle.load(fp)

        df_corr, low_week_sampels_dict = d.weekly_correlation(df_weekly_mean=df_weekly,
                                                              tickers_list=clean_ticker_list,
                                                              # tickers_list=new_list,
                                                              min_prcnt_data=min_prcnt_data,
                                                              threshold=min_days_in_week,
                                                              path_from='weekly_diff',
                                                              set_as_index=['first_day_in_week', 'last_day_in_week']
                                                              )

        start_time = time.time()
        df_reg_full = d.reg_df(
            ticker_name=main_ticker,
            df_weekly=df_weekly,
            df_corr=df_corr,
            start_date=date_reference,
            threshold=corr_threshold,
            # activate_automated_rolling=True
        )

        start_time = time.time()
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

        train_dict_c = train.df_filtered_dict

        train = Trainn(NAME,
                       df_reg=df_reg,
                       test_size=test_size,
                       path=models_path)

        train_dict_p = train.df_filtered_dict

        dd['tick'].append(n)

        dd['rmse_train_c'].append(train_dict_c['rmse_train'])
        dd['rmse_train_p'].append(train_dict_p['rmse_train'])
        dd['rmse_train_diff'].append(train_dict_c['rmse_train'] - train_dict_p['rmse_train'])

        dd['rmse_test_c'].append(train_dict_c['rmse_test'])
        dd['rmse_test_p'].append(train_dict_p['rmse_test'])
        dd['rmse_test_diff'].append(train_dict_c['rmse_test'] - train_dict_p['rmse_test'])

        dd['r2_test_c'].append(train_dict_c['r2_test'])
        dd['r2_test_p'].append(train_dict_p['r2_test'])
        dd['r2_test_diff'].append(train_dict_c['r2_test'] - train_dict_p['r2_test'])

        dd['r2_train_c'].append(train_dict_c['r2_train'])
        dd['r2_train_p'].append(train_dict_p['r2_train'])
        dd['r2_train_diff'].append(train_dict_c['r2_train'] - train_dict_p['r2_train'])

        dd['pred_num_c'].append(train_dict_c['predictor_num'])
        dd['pred_num_p'].append(train_dict_p['predictor_num'])
        dd['pred_num_diff'].append(train_dict_c['predictor_num'] - train_dict_p['predictor_num'])

        df = pd.DataFrame(dd)
        print(df)
        df.to_csv('dev_test_p_vs_c.csv')
    except:
        print(' -- filed -- ' * 10, len(tickers) - j, '    =====   ', n, 'df_reg_shape', df_reg_full.shape)
