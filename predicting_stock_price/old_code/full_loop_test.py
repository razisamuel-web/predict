from feature_engineering import featureEng
from data_generation import get_data
from evaluator import Evaluator
from train import Train
import pandas as pd
from predict import Predict
from datetime import datetime
from general_functions import corrs_mean_from_cols_names, next_week_full_train_behavior
import pickle

fl_dict = {}

ticker_name_l = []
years_back_l = []
df_raw_shape_l = []
df_clean_shape_l = []
df_feacheng_shape_l = []
df_dailydiff_shape_l = []
df_weeklydiff_shape_l = []
threshold_days_in_week_l = []
df_corr_shape_l = []
min_prcnt_data_l = []
df_reg_shape_l = []
predictors_number_l = []
predictors_names_corr_l = []
corrs_l = []
corrs_mean_l = []
corr_threshold_l = []
test_size_l = []
X_train_l = []
X_test_l = []
y_train_l = []
y_test_l = []
rmse_train_l = []
r2_train_l = []
rmse_test_l = []
r2_test_l = []
alpha_l = []
given_date_l = []
exist_date_reference_l = []
last_close_price_l = []
predicted_diff_l = []
next_week_price_l = []
percentage_change_l = []

rmse_train_full_train_l = []
r2_train_full_train_l = []
alpha_full_train_l = []
predicted_diff_full_train_l = []
next_week_price_full_train_l = []
percentage_change_full_train_l = []

# %%
with open("clean_ticker_list.txt", "rb") as fp:  # Unpickling
    tickers = pickle.load(fp)

# %%
c = 0
for tick in tickers:
    c += 1
    try:
        main_ticker = tick
        years_back = 1 #1.5
        threshold_days_in_week = 1
        min_prcnt_data = 0.7
        corr_threshold = 0.34 # 0.3
        test_size = 0.1

        # before after validation
        dfr = get_data(ticker_name=main_ticker, data_from_csv=1, path_from='raw_data')
        dfc = get_data(ticker_name=main_ticker, data_from_csv=1, path_from='data')

        # feature engineering
        d = featureEng(main_ticker, years_back=years_back, data_from_csv=True, path_from='data')
        df = d.daily_diff()
        df_weekly = d.weekly_mean(df, threshold_days_in_week=threshold_days_in_week)
        df_corr = d.weekly_correlation(df_weekly_mean=df_weekly, tickers_list=tickers, min_prcnt_data=min_prcnt_data)
        df_reg = d.reg_df(df_weekly, df_corr, threshold=corr_threshold)

        # train
        target_column_name = f'{main_ticker}_main_symb_weekly_mean_diff'
        train = Train(target_column_name)
        X_train, X_test, y_train, y_test = train.split_to_train_test(df_reg, test_size=0.1, random_state=3)
        reg = train.fittrain(X_train=X_train, y_train=y_train)

        # evaluation
        evaluator = Evaluator(ticker=main_ticker, reg=reg, X_train=X_train, X_test=X_test, y_test=y_test,
                              y_train=y_train)
        ev_d = evaluator.statistical_metrics()

        # predict
        predict = Predict(reg, target_column_name, df_reg=df_reg)
        pr_d = predict.next_week_behavior(df=d._df, date_reference=datetime.today())

        # next_week_full_train_behavior
        pr_d_f_t = next_week_full_train_behavior(main_ticker=main_ticker, df_reg=df_reg, df_row=d._df,
                                                 date_reference=datetime.today())

        # results
        df_feacheng_shape_l.append(str(d._df.shape))
        df_dailydiff_shape_l.append(str(df.shape))
        df_weeklydiff_shape_l.append(str(df_weekly.shape))
        threshold_days_in_week_l.append(threshold_days_in_week)
        df_corr_shape_l.append(str(df_corr.shape))
        min_prcnt_data_l.append(min_prcnt_data)
        df_reg_shape_l.append(str(df_reg.shape))
        predictors_number_l.append(df_reg.shape[1])
        predictors_names_corr_l.append(df_reg.columns)
        corrs, corrs_mean = corrs_mean_from_cols_names(df_reg.columns)
        corrs_l.append(corrs)
        corrs_mean_l.append(corrs_mean)
        corr_threshold_l.append(corr_threshold)
        test_size_l.append(test_size)
        X_train_l.append(len(X_train))
        X_test_l.append(len(X_test))
        y_train_l.append(len(y_train))
        y_test_l.append(len(y_test))
        rmse_train_l.append(ev_d['rmse_train'])
        r2_train_l.append(ev_d['r2_train'])
        rmse_test_l.append(ev_d['rmse_test'])
        r2_test_l.append(ev_d['r2_test'])
        alpha_l.append(ev_d['alpha'])
        given_date_l.append(pr_d['given_date'][0])
        exist_date_reference_l.append(pr_d['exist_date_reference'][0])
        last_close_price_l.append(pr_d['last_close_price'][0])
        predicted_diff_l.append(pr_d['predicted_diff'][0])
        next_week_price_l.append(pr_d['next_week_price'][0])
        percentage_change_l.append(pr_d['percentage_change'][0])

        alpha_full_train_l.append(pr_d_f_t['alpha_full_train'])
        rmse_train_full_train_l.append(pr_d_f_t['rmse_train_full_train'])
        r2_train_full_train_l.append(pr_d_f_t['r2_train_full_train'])
        predicted_diff_full_train_l.append(pr_d_f_t['predicted_diff_full_train'])
        next_week_price_full_train_l.append(pr_d_f_t['next_week_price_full_train'])
        percentage_change_full_train_l.append(pr_d_f_t['percentage_change_full_train'])

        ticker_name_l.append(main_ticker)
        years_back_l.append(years_back)
        df_raw_shape_l.append(str(dfr.shape))
        df_clean_shape_l.append(str(dfc.shape))

        print(f"""round number {len(tickers) - c},  tick ={tick}, r^2 test ={ev_d["r2_test"]}, alpha = {ev_d["alpha"]}',
              'predicted_diff' ={pr_d["predicted_diff"][0]}, percentage_change = {pr_d['percentage_change'][0]}""")
    except:
        print(f'check {tick}, df_reg_shape = {df_reg.shape}')

# %%
fl_dict = {'ticker_name': ticker_name_l,
           'years_back': years_back_l,
           'df_raw_shape': df_raw_shape_l,
           'df_clean_shape': df_clean_shape_l,
           'df_feacheng_shape': df_feacheng_shape_l,
           'df_dailydiff_shape': df_dailydiff_shape_l,
           'df_weeklydiff_shape': df_weeklydiff_shape_l,
           'threshold_days_in_week': threshold_days_in_week_l,
           'df_corr_shape': df_corr_shape_l,
           'min_prcnt_data': min_prcnt_data_l,
           'df_reg_shape': df_reg_shape_l,
           'predictors_number': predictors_number_l,
           'predictors_names_corr': predictors_names_corr_l,
           'corrs': corrs_l,
           'corrs_mean': corrs_mean_l,
           'corr_threshold': corr_threshold_l,
           'test_size': test_size_l,
           'X_train': X_train_l,
           'X_test': X_test_l,
           'y_train': y_train_l,
           'y_test': y_test_l,

           'rmse_train': rmse_train_l,
           'rmse_train_full_train': rmse_train_full_train_l,

           'r2_train': r2_train_l,
           'r2_train_full_train': r2_train_full_train_l,

           'rmse_test': rmse_test_l,
           'r2_test': r2_test_l,

           'alpha': alpha_l,
           'alpha_full_train': alpha_full_train_l,

           'given_date': given_date_l,
           'exist_date_reference': exist_date_reference_l,
           'last_close_price': last_close_price_l,

           'predicted_diff': predicted_diff_l,
           'predicted_diff_full_train': predicted_diff_full_train_l,

           'next_week_price': next_week_price_l,
           'next_week_price_full_train_': next_week_price_full_train_l,

           'percentage_change': percentage_change_l,
           'percentage_change_full_train': percentage_change_full_train_l

           }
# %%
df_summary = pd.DataFrame(fl_dict)
df_summary = df_summary.sort_values(by=['alpha', 'percentage_change', 'r2_test', 'rmse_test'])
name = './results/summary_30_05_2020_corr_0_34.csv'
df_summary.to_csv(name)
# %%
df_new = df_summary[['ticker_name', 'alpha', 'percentage_change', 'r2_test', 'rmse_test']]
