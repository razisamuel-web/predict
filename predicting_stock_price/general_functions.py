import re
import numpy as np
from evaluator import Evaluator
from predict import Predict
from datetime import datetime, timedelta
from train import ClfTrain
# from feature_engineering_roll import featureEng
from feature_engineering import featureEng
import pandas as pd
from data_generation import save_as_csv, get_data
from sklearn import linear_model
import copy
import os


def corrs_mean_from_cols_names(string_list):
    '''
    'def corrs_mean_from_cols_names' calculate the mean of the predixtors correlation
    :param string_list: columns names, each column contain stock name and corr value
    :return: list of corrs and mean of the list
    '''
    p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    corrs = []
    for i in string_list:
        s = i
        if re.search(p, s) is not None:
            for catch in re.finditer(p, s):
                corrs.append(float(catch[0]))
    return corrs, round(np.mean(corrs), 2)


def next_week_full_train_behavior(main_ticker, df_reg_full, df_raw, cols, train_object, clftrain_object,
                                  days_interval,
                                  date_reference):
    '''
    next_week_full_train_behavior learn and predict stock performence with respect tot all data
    :param main_ticker: str, currnet stock name
    :param df_reg: pandas data frame
    :param df_row: pandas data frame
    :return: dictionary of statistical metrics and preddictions
    '''

    target_column_name = f'{main_ticker}_main_symb_weekly_mean_diff'
    df_reg = copy.copy(df_reg_full[:-1])
    X_train, y_train = df_reg.drop([target_column_name], axis=1).to_numpy(), \
                       df_reg[target_column_name].to_numpy()
    model_params = train_object.reg.best_params_
    model_params = {i.replace('model__', ''): j for i, j in model_params.items()}
    reg = train_object.model(**model_params)
    reg.fit(X_train, y_train)

    predict = Predict(reg, target_column_name, df_reg=df_reg_full, date_reference=date_reference, cols=cols,
                      days_interval=days_interval)
    pr_f_t = predict.next_week_behavior(df=df_raw, date_reference=date_reference)

    from evaluator import Evaluator

    evaluator = Evaluator(ticker=main_ticker,
                          reg=reg,
                          X_train=X_train,
                          X_test=X_train[0:2], y_train=y_train,
                          y_test=y_train[0:2])

    ev_d_f_t = evaluator.statistical_metrics()

    cls_cv_best_params = clftrain_object._lr_clf.best_params_
    cls_model = clftrain_object.learn_with_chosen_params(cls_cv_best_params=cls_cv_best_params,
                                                         x=X_train,
                                                         y=y_train)

    next_week_class = predict.next_week_class(cls_model)

    r = {'rmse_train_full_train': ev_d_f_t['rmse_train'],
         'r2_train_full_train': ev_d_f_t['r2_train'],
         'alpha_full_train': model_params['alpha'],
         'predicted_diff_full_train': pr_f_t['predicted_diff'][0],
         'next_week_price_full_train': pr_f_t['next_week_price'][0],
         'percentage_change_full_train': pr_f_t['percentage_change'][0]}

    r = {k: round(v, 4) for k, v in r.items()}

    r.update(next_week_class)

    return r


def weekly_correlations_to_csv(tickers_list, years_back_data_generation, start_date, days_interval, threshold_days_in_week,
                               path='./weekly_diff'):
    c = 0
    if not os.path.exists(path):
        os.mkdir(path)
    new_tickers_list = []
    for tick in tickers_list:
        if c % 40 == 0:
            print(len(tickers_list) - c)
        c += 1
        tick_ta = tick + '.TA'

        dd_feat_eng = featureEng(tick_ta, date_reference=start_date, years_back=years_back_data_generation)
        df = dd_feat_eng.daily_diff()

        df = dd_feat_eng.weekly_mean(ticker_name=tick_ta,
                                     df=df,
                                     start_date=start_date,
                                     days_interval=days_interval,
                                     threshold_days_in_week=threshold_days_in_week
                                     )

        save_as_csv(tick, df, outdir=path)

        new_tickers_list.append(tick)

    return new_tickers_list


def reindex(dfinx):
    new_index = []

    for e in dfinx:
        if ('_1' in e) and len(e) == 6:
            new_date = f'{int(e[:4]) - 1}_52'
            new_index.append(new_date)
        else:
            new_index.append(f'{e[:5]}{int(e[5:]) - 1}')

    return new_index


def create_weekly_mean(df_clean):
    df_clean['dayofweek'] = df_clean.index.dayofweek
    df_clean['weekly_mean'] = None
    sm = 0
    jj = 0
    s = df_clean.index[0]
    for d, i, j in zip(df_clean.index, df_clean['close'], df_clean['dayofweek']):
        if jj > j:
            avg = sm / jj

            df_clean.loc[s:e, 'weekly_mean'] = avg
            jj = 1
            sm = i
            s = d
        else:
            e = d
            sm += i
            jj += 1

    avg = sm / jj
    e = d
    df_clean.loc[s:e, 'weekly_mean'] = avg
    df_clean = df_clean.iloc[1:, :]
    return df_clean


def predictor_df_for_plot(tick_name, df_reg_columns, dfc):
    predictor_names = [re.sub(r"[^A-Z]", '', s.replace('.TA', '_')) for s in df_reg_columns if 'main' not in s]
    # TODO PAR OF THE TICKER NAMES CONTAIN NUMBERS AS 'RIT1.TA' FIX THE FUNCTION TO SUPPPORT IS INSTEAD USING "TRY"
    predictor_names = [s.split('INT') for s in predictor_names] + [[tick_name]]

    df_predictor = copy.copy(dfc[['close']])
    for tick_n in predictor_names:
        try:
            if len(tick_n) == 1:
                tick_n = tick_n[0]
                a = get_data(ticker_name=tick_n, data_from_csv=1, path_from='data')
                if tick_n == tick_name:
                    a.index = a.index - timedelta(weeks=1)
                df_predictor.loc[:, tick_n] = a['close']

            if len(tick_n) == 2:
                a = get_data(ticker_name=tick_n[0], data_from_csv=1, path_from='data')
                b = get_data(ticker_name=tick_n[1], data_from_csv=1, path_from='data')
                df_predictor.loc[:, '_INT_'.join(tick_n)] = a['close'] * b['close']
        except:
            print(f'fail recover {tick_n}')

    df_predictor.drop(['close'], axis=1, inplace=True)
    return df_predictor


def top_corrs(tick_name, df, top_x):
    df_predictor = df[:]
    df_corrs = df_predictor.corr()
    df_predictor.columns = [f'{j}_{round(i, 2)}' for i, j in zip(df_corrs[tick_name], df_corrs.index)]
    abs_corrs = list(abs(df_corrs[tick_name]))
    X = df_predictor.columns
    Y = abs_corrs
    Z = [x for _, x in sorted(zip(Y, X))]
    top_n_l = Z[(len(Z) - top_x):len(Z)]
    df = df_predictor[top_n_l]
    return df


def min_max_normalization(df):
    df = df / (df.max() - df.min())
    return df
