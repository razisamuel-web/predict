from feature_engineering import featureEng
from data_generation import get_data
from train import Train, ClfTrain
import pandas as pd
from predict import Predict
from general_functions import corrs_mean_from_cols_names, next_week_full_train_behavior
import pickle
import time
import copy
import os

#
full_loop_path = ['df_regs_full_loop', 'full_df_regs_loop']
for i in full_loop_path:
    if not os.path.exists(i):
        os.mkdir(i)

# %%
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
low_week_samples_dict_l = []
df_reg_shape_l = []
predictors_number_l = []
predictors_names_corr_l = []
corrs_l = []
corrs_mean_l = []
corr_threshold_l = []
corr_inter_threshold_l = []
inter_columns_l = []
number_of_inter_l = []
min_days_in_week_threshold_l = []
test_size_l = []
X_train_l = []
X_test_l = []
y_train_l = []
y_test_l = []

rmse_train_l = []
r2_train_l = []
r2_adj_train_l = []

rmse_test_l = []
r2_test_l = []
r2_adj_test_l = []

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

lr_class_accuracy_test_l = []
next_week_class_l = []
next_week_class_fp_l = []
next_week_class_tp_l = []
next_week_class_full_train_l = []
next_week_class_fp_full_train_l = []
next_week_class_tp_full_train_l = []

df_reg_time_l = []
df_reg_inter_time_l = []
train_time_l = []
predict_time_l = []
full_train_time_l = []
save_results_time_l = []
run_time_l = []
# %%
# TODO:TOPPIS = min_days_in_week, low_week_samples_dict, WEEKLY_DIFF_FROM_CSV

with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
    tickers = pickle.load(fp)

# %%

tickers_to_run = tickers
tickers_to_corr = tickers

test_size = 0.15
years_back = 1.4
threshold_days_in_week = 1
min_prcnt_data = 0.8
min_days_in_week = 1
corr_threshold = 0.35
corr_inter_threshold = 0.7
days_interval = 6
date_reference = '2020-09-12'
models_path = './pickel_models'
corr_inter_threshold_main = corr_threshold

c = 0
errors = {}

for tick in tickers_to_run:
    c += 1

    print('\n=  -  =  -  =  -  =  -  =  -  =  -  =  -  =  ', tick, '   =  -  =  -  =  -  =  -  =  -  =  -  =  -  =')
    try:
        start_start = time.time()
        main_ticker = tick

        # before after validation
        dfr = get_data(ticker_name=main_ticker, data_from_csv=1, path_from='raw_data')
        dfc = get_data(ticker_name=main_ticker, data_from_csv=1, path_from='data')

        # feature engineering
        d = featureEng(main_ticker,
                       years_back=years_back,
                       data_from_csv=True,
                       path_from='data')

        df = d.daily_diff()

        df_weekly = get_data(main_ticker,
                             data_from_csv=True,
                             path_from='weekly_diff',
                             set_as_index=['first_day_in_week', 'last_day_in_week'],
                             index_type='object')

        df_corr, low_week_samples_dict = d.weekly_correlation(df_weekly_mean=df_weekly,
                                                              tickers_list=tickers_to_corr,
                                                              min_prcnt_data=min_prcnt_data,
                                                              threshold=min_days_in_week,
                                                              path_from='weekly_diff',
                                                              set_as_index=['first_day_in_week',
                                                                            'last_day_in_week']
                                                              )

        df_reg_full = d.reg_df(ticker_name=main_ticker,
                               df_weekly=df_weekly,
                               df_corr=df_corr,
                               start_date=date_reference,
                               threshold=corr_threshold)

        df_reg_time = round(time.time() - start_start, 1)
        start = time.time()

        df_reg_full = d.df_reg_int(df_reg=df_reg_full,
                                   corr_inter_threshold=corr_inter_threshold,
                                   corr_inter_threshold_main=corr_inter_threshold_main)

        df_reg = copy.copy(df_reg_full[:-1])

        df_reg_inter_time = round(time.time() - start, 1)

        start = time.time()

        inter_columns = [inter for inter in df_reg.columns if 'INT' in inter]
        number_of_inter = len(inter_columns)

        # train
        # -- continues
        target_column_name = f'{main_ticker}_main_symb_weekly_mean_diff'

        train = Train(main_ticker,
                      df_reg=df_reg,
                      test_size=test_size,
                      path=models_path)

        # -- class
        clftrain = ClfTrain(tick_name=main_ticker)
        clf = clftrain.fit_lr_gridsearch_cv()
        summary_dict = clftrain.generate_clf_summary(clf, classifire_type='lr')

        train_time = round(time.time() - start, 1)
        start = time.time()

        train_dict = train.df_filtered_dict
        reg = train.reg

        # saves
        colsl = train_dict['current_corrs_str']
        df_reg = df_reg[colsl]
        df_reg.to_csv(f'./df_regs_full_loop/df_reg_{tick}.csv')
        df_reg_full = df_reg_full[colsl]
        df_reg_full.to_csv(f'./full_df_regs_loop/df_reg_{tick}.csv')

        # predict
        # -- continues
        predict = Predict(reg,
                          target_column_name,
                          df_reg=df_reg_full,
                          date_reference=date_reference,
                          cols=colsl,
                          days_interval=days_interval)

        pr_d = predict.next_week_behavior(df=d._df,
                                          date_reference=date_reference)
        predict_time = round(time.time() - start, 1)

        # -- class
        next_week_class_dict = predict.next_week_class(clf)

        start = time.time()

        # next_week_full_train_behavior
        pr_d_f_t = next_week_full_train_behavior(main_ticker=main_ticker,
                                                 df_reg_full=df_reg_full,
                                                 df_raw=d._df,
                                                 cols=colsl,
                                                 train_object=train,
                                                 clftrain_object=clftrain,
                                                 days_interval=days_interval,
                                                 date_reference=date_reference
                                                 )

        full_train_time = round(time.time() - start, 1)

        start = time.time()
        # results
        df_reg_time_l.append(df_reg_time)
        df_reg_inter_time_l.append(df_reg_inter_time)
        train_time_l.append(train_time)
        predict_time_l.append(predict_time)
        full_train_time_l.append(full_train_time)
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
        corr_inter_threshold_l.append(corr_inter_threshold)
        inter_columns_l.append(inter_columns)
        number_of_inter_l.append(number_of_inter)
        min_days_in_week_threshold_l.append(min_days_in_week)
        low_week_samples_dict_l.append(low_week_samples_dict)
        test_size_l.append(test_size)
        X_train_l.append(round(df_reg.shape[0] * (1 - test_size)))
        X_test_l.append(round(df_reg.shape[0] * test_size))

        rmse_train_l.append(train_dict['rmse_train'])
        r2_train_l.append(train_dict['r2_train'])
        rmse_test_l.append(train_dict['rmse_test'])
        r2_test_l.append(train_dict['r2_test'])
        alpha_l.append(train_dict['alpha'])
        r2_adj_train_l.append(train_dict['r2_ad_nps_train'])
        r2_adj_test_l.append(train_dict['r2_ad_nps_test'])

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

        lr_class_accuracy_test_l.append(round(summary_dict['accuracy_test'], 2))
        next_week_class_l.append(next_week_class_dict['class'])
        next_week_class_fp_l.append(round(next_week_class_dict['False_p'], 2))
        next_week_class_tp_l.append(round(next_week_class_dict['True_p'], 2))

        next_week_class_full_train_l.append(pr_d_f_t['class'])
        next_week_class_fp_full_train_l.append(round(pr_d_f_t['False_p'], 2))
        next_week_class_tp_full_train_l.append(round(pr_d_f_t['True_p'], 2))

        ticker_name_l.append(main_ticker)
        years_back_l.append(years_back)
        df_raw_shape_l.append(str(dfr.shape))
        df_clean_shape_l.append(str(dfc.shape))
        save_results_time = time.time() - start
        save_results_time_l.append(save_results_time)
        run_time = round(time.time() - start_start, 1)
        run_time_l.append(run_time)

        print(
            f""" --- round number {len(tickers_to_run) - c} ---\ntick ={tick}, r^2 test ={round(train_dict[
                                                                                                    "r2_test"],
                                                                                                2)}, alpha = {
            train_dict[
                "alpha"]}',\n --  'predicted_diff' ={round(pr_d["predicted_diff"][0], 2)}, percentage_change = {
            pr_d['percentage_change'][
                0]}  -- \npredictor_number = {df_reg.shape[
                1]} ,\nrun_time = {run_time}, reg_time= {df_reg_time}, inter_time = {df_reg_inter_time
            }, train_time = {train_time} ,full_train_time = {full_train_time}""")

        del dfc, dfr, df_corr, df_weekly, df, d, train, pr_d_f_t

        fl_dict = {'ticker_name': ticker_name_l,
                   'df_reg_time': df_reg_time_l,
                   'df_reg_inter_time': df_reg_inter_time_l,
                   'train_time': train_time_l,
                   'predict_time': predict_time_l,
                   'full_train_time': full_train_time_l,
                   'save_results_time': save_results_time_l,
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
                   #  'predictors_names_corr': predictors_names_corr_l,
                   #  'corrs': corrs_l,
                   'corrs_mean': corrs_mean_l,
                   'corr_threshold': corr_threshold_l,
                   'corr_inter_threshold': corr_inter_threshold_l,
                   #  'inter_columns': inter_columns_l,
                   'number_of_inter': number_of_inter_l,

                   'min_days_in_week_threshold': min_days_in_week_threshold_l,
                   'low_week_samples_dict': low_week_samples_dict_l,

                   'test_size': test_size_l,
                   'X_train_shape': X_train_l,
                   'X_test_shape': X_test_l,

                   'rmse_train': rmse_train_l,
                   'rmse_train_full_train': rmse_train_full_train_l,

                   'r2_train': r2_train_l,
                   'r2_train_full_train': r2_train_full_train_l,

                   'rmse_test': rmse_test_l,
                   'r2_test': r2_test_l,

                   'r2_ad_nps_train': r2_adj_train_l,
                   'r2_ad_nps_test': r2_adj_test_l,

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
                   'percentage_change_full_train': percentage_change_full_train_l,

                   'lr_class_accuracy_test': lr_class_accuracy_test_l,
                   'next_week_class': next_week_class_l,
                   'next_week_class_fp': next_week_class_fp_l,
                   'next_week_class_tp': next_week_class_tp_l,
                   'next_week_class_full_train': next_week_class_full_train_l,
                   'next_week_class_fp_full_train': next_week_class_fp_full_train_l,
                   'next_week_class_tp_full_train': next_week_class_tp_full_train_l

                   }

        df_summary = pd.DataFrame(fl_dict)
        df_summary = df_summary.sort_values(by=['alpha', 'percentage_change', 'r2_test', 'rmse_test'])

        df_summary.to_csv('./results/2020-08-01.csv')

    except:
        try:
            if df_reg.shape[1] > 1:
                errors[tick] = f'df_reg.shape[1]= {df_reg.shape[1]}'
            if (str(max(dfc.index)) < str(date_reference)):
                errors[tick] = f'{tick}, max date  = {max(dfc.index)} < date reference = {date_reference}'

        except:
            print(tick, '=====' * 100)

# %%
df_new = df_summary[['ticker_name', 'alpha', 'percentage_change', 'r2_test', 'rmse_test']]

# %%


# problematic = ABIL,APLY,ARNA,FIBI,FBRT,GAON,GIBUI,GILT,GNGR,GODM-M,HDST,HGG,IDIN,ICL,INRM,LCTX
