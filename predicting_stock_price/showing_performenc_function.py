import pandas as pd
from IPython.display import display
from data_generation import get_data
from general_functions import create_weekly_mean, predictor_df_for_plot, top_corrs, min_max_normalization
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import pickle
from regressors import stats


pd.set_option("display.max_columns", 101)


def result_summary(tick_name, cnt=70, path = './results/2020-06-14_summary.csv'):
    df_problematic_path = './df_problematic/df_problematic_open.csv'
    df_problematic = pd.read_csv(df_problematic_path)

    df_results_path = path
    df_results = pd.read_csv(df_results_path)

    df_problematic_tick_properties_df = df_problematic[df_problematic['ticker_name'] == tick_name]
    print('\n', '-- problematic data frame  --'.center(cnt, '='), '\n')
    display(df_problematic_tick_properties_df)

    print('\n', '-- statistical results  -- '.center(cnt, '='), '\n')
    df_tick_results = df_results[df_results['ticker_name'] == tick_name]
    display(df_tick_results)

    main_metrics_technical = [
        'df_reg_shape',
        'corrs_mean',
        'number_of_inter',
        'given_date',
        'last_close_price', ]

    df_tick_results_technical = df_tick_results[main_metrics_technical]
    print('\n', '-- technical   --'.center(cnt, ' '), '\n')
    display(df_tick_results_technical)

    main_metrics_statistical = ['rmse_train', 'r2_train', 'r2_train_full_train',
                                'rmse_test', 'r2_test', 'alpha', 'alpha_full_train',
                                'predicted_diff',
                                'predicted_diff_full_train', 'next_week_price',
                                'next_week_price_full_train_', 'percentage_change',
                                'percentage_change_full_train']

    main_metrics_statistical = df_tick_results[main_metrics_statistical]
    print('\n', '-- stastics   --'.center(cnt, ' '), '\n')
    display(main_metrics_statistical)


def clean_raw_and_weekly_mean_plot(tick_name):
    dfc = get_data(ticker_name=tick_name, data_from_csv=1, path_from='data')
    dfc = create_weekly_mean(dfc)
    dfp = dfc[['close', 'weekly_mean']]
    dfp.plot(style='o-', ms=2)
    plt.show()

    max_date = max(dfp.index)
    max_date_lower = max_date - timedelta(weeks=4)
    dfp_last = dfp.loc[max_date_lower:max_date, :]
    dfp_last.plot(style='o-', ms=2)
    plt.show()


def correlated_ticks_plot(tick_name, top, weeks_back: list, df_reg_summary):
    '''
    The following function plot the top correlated and sagnificant features
    :param tick_name: current tick_name
    :param top: top correlated or to p_values
    :param weeks_back: number
    :param df_reg_summary: pandas data frame
    :return: plots
    '''
    full_df_reg = pd.read_csv(f'./full_df_regs_loop/df_reg_{tick_name}.csv', index_col=['first_day_in_week', 'last_day_in_week'])
    full_df_reg.sort_index(axis=0, inplace=True)
    full_df_reg.index = [f'{i[0][2:]}\n{i[1][5:]}' for i in full_df_reg.index]
    df_reg = full_df_reg[:]

    df_reg_plot = df_reg[:]
    df_reg_plot.columns = [i.replace('.TA_weekly_mean_diff', '') for i in df_reg.columns]
    # df_reg_plot = (df_reg_plot - df_reg_plot.mean()) / df_reg_plot.std()

    df_reg_plot = min_max_normalization(df_reg_plot)

    main_column = [i for i in df_reg_plot.columns if 'main' in i]
    corrs_df_reg = [float(i[len(i) - 3: len(i)]) for i in df_reg_plot.columns if 'main' not in i]

    # corrs section
    X = df_reg_plot.columns[1:]
    Y = corrs_df_reg
    Z = [x for _, x in sorted(zip(Y, X))][::-1]
    Z = main_column + Z[0:top + 1]
    df_reg_plot_ordered = df_reg_plot[Z]
    df_reg_plot_ordered.plot(style='o-', ms=3, )
    plt.legend(loc=2, fontsize='x-small')
    plt.show()

    for d in weeks_back:
        n = df_reg.shape[0]
        df_reg_plot_ordered_last_weeks = df_reg_plot_ordered.iloc[n - d:n, :]
        df_reg_plot_ordered_last_weeks.plot(style='o-', ms=3, )
        plt.legend(loc=2, fontsize='x-small')
        plt.show()
    # p_values section
    d = df_reg_summary.index + '_B_' + df_reg_summary['Estimate'].round(3).astype(str) + '_p_value_' + df_reg_summary[
        'p value'].round(3).astype(str)
    d = pd.DataFrame(d)

    X = df_reg_plot.columns[1:]
    Y = df_reg_summary['p value'][1:]
    Z = [x for _, x in sorted(zip(Y, X))][::-1]
    Z = main_column + Z[0:top + 1]
    df_reg_plot_ordered = df_reg_plot[Z]

    df_reg_plot_ordered.columns = [df_reg_plot_ordered.columns[0]] + [d.loc[i, :][0] for i in
                                                                      df_reg_plot_ordered.columns if
                                                                      i in df_reg_summary.index]
    for d in weeks_back:
        n = df_reg.shape[0]
        df_reg_plot_ordered_last_weeks = df_reg_plot_ordered.iloc[n - d:n, :]
        df_reg_plot_ordered_last_weeks.plot(style='o-', ms=3, )

        plt.legend(loc=2, fontsize='x-small')
        plt.show()


def daily_correlated_ticks_plot(tick_name, df_reg_columns, weeks_back):
    # all weeks
    dfc = get_data(ticker_name=tick_name, data_from_csv=1, path_from='data')
    df_predictor = predictor_df_for_plot(df_reg_columns=df_reg_columns, dfc=dfc, tick_name=tick_name)
    df = top_corrs(tick_name=tick_name, df=df_predictor, top_x=10)

    df = df/df.max()
    df.plot(style='o-', ms=2)
    plt.legend(loc=2, fontsize='x-small')
    plt.show()

    # last_week

    max_date = max(df.index)
    max_date_lower = max_date - timedelta(weeks=weeks_back)
    dfp_last = df.loc[max_date_lower:max_date, :]
    dfp_last.plot(style='o-', ms=3, )
    plt.legend(loc=2, fontsize='x-small')
    plt.show()


def generate_df_reg_summary(tick_name, reg, df_reg_columns):
    a_file = open(f"./df_trained_filtered_dict/{tick_name}.pkl", "rb")
    output = pickle.load(a_file)
    y_train = output['y_train']
    x_train = output['x_train']

    resid, r2, f, df_reg_summary = stats.summary(reg, x_train, y_train)
    coefs_names = [i.replace('.TA_weekly_mean_diff', '') for i in df_reg_columns]
    coefs_names = ['intercept'] + coefs_names[1:]
    df_reg_summary.index = coefs_names
    return resid, r2, f, df_reg_summary


def get_rolling_mean_std(df_raw_close: pd.DataFrame, df_daily_diff: pd.DataFrame, date_reference: str,
                         mean_days_roll: int, std_days_roll: int, show_days_range_back: int):
    '''
    thisf function gives general information on interested stock, last days means and standard deviation
    :param df_raw_close: pd.Dataframe, as same as downloaded from yahoo finance (after cleaning )
    :param df_daily_diff: pd.Dataframe, diffs of each day with the day before
    :param date_reference: string, point out on interested date
    :param mean_days_roll: integer, number of days for mean
    :param std_days_roll: integer, number of days for std
    :param show_days_range_back: integer, number of days to show from date reference
    :return: pd.DataFrame with all the statistics:['close', 'mean_close', 'std_close', 'diff_stock_name', 'mean_diff_stock_name',
       'std_diff_stock_name']
    '''
    max_range = max([mean_days_roll, std_days_roll, show_days_range_back]) + 1
    end_date = datetime.strptime(date_reference, '%Y-%m-%d')
    start_date = str(end_date - timedelta(days=max_range * 2 + 1))

    df_raw_close = df_raw_close.loc[start_date:end_date, :]
    df_daily_diff = df_daily_diff.loc[start_date:end_date, :]

    df_rolled = df_raw_close[:]
    for df in (df_raw_close, df_daily_diff):
        col = df.columns[0]
        df_rolled[col] = df[col]
        df_rolled[f'mean_{col}'] = df[col].rolling(mean_days_roll, min_periods=1).mean()
        df_rolled[f'std_{col}'] = df[col].rolling(std_days_roll, min_periods=1).std(ddof=0)

    end_date = datetime.strptime(date_reference, '%Y-%m-%d')
    start_date = str(end_date - timedelta(days=show_days_range_back))
    df_statistics = df_rolled.loc[start_date:end_date, :]

    return df_statistics