from yahoo_fin import stock_info as si
from datetime import date, timedelta, datetime
from data_generation import get_data
import numpy as np
import pandas as pd
import copy


# TODO : adding warning in case of main ticker has no data from last day warnings.warn("blabla", DeprecationWarning)
# TODO : remove dont remove 'years_back'

class featureEng(object):

    def __init__(self, ticker_name, years_back=1, data_from_csv=True, path_from='data'):
        self._path_from = path_from
        self._years_back = years_back
        self._today = str(date.today())
        self._same_date_last_year = str(date.today() - timedelta(days=round(self._years_back * 365)))
        self._ticker_name = ticker_name
        self._data_from_csv = data_from_csv
        self._df_befor_reparing = get_data(ticker_name, data_from_csv, path_from=self._path_from)
        self._df = self.repair_open_close(self._df_befor_reparing)
        self._rows_indexs_na_values = self._nan_indexs_search(self._df)
        self._df = self._nan_remover(self._df, self._rows_indexs_na_values)
        self._df_daily_diff_open_close = self._daily_diff_open_close()

    def repair_open_close(self, df):
        open = df['open'][0]
        dfc_open = df[['close']]
        dfc_open.index = [i for i in df.index[1:]] + [1]
        df['open'] = dfc_open['close']
        df.iloc[0:1, 0:1] = open
        return df

    def _daily_diff_open_close(self):

        df = self._df.drop(['high', 'low', 'adjclose', 'volume', 'ticker', ], axis=1)
        # df[f'diff_{self._ticker_name}'] = (df['close'] - df['open']) / df['open']
        df[f'diff_{self._ticker_name}'] = df['close'] - df['open']

        return df

    def daily_diff(self):

        df = self._df.drop(['high', 'low', 'adjclose', 'volume', 'ticker', ], axis=1)
        # df[f'diff_{self._ticker_name}'] = (df['close'] - df['open']) / df['open']
        df[f'diff_{self._ticker_name}'] = df['close'] - df['open']

        df_daily_diff_open_clos = df
        df.drop(['open', 'close', ], axis=1, inplace=True)

        return pd.DataFrame(df[f'diff_{self._ticker_name}'])

    def weekly_mean(self, df, threshold_days_in_week=2):
        df['dayofweek'] = df.index.dayofweek
        df['year'] = df.index.year
        df['original_date'] = df.index
        df = self._fix_first_day_of_week(df)

        year_week = self.index_to_year_week(df.index)
        df['week_of_year'] = year_week

        d_weekly_mean = {}
        d_weekly_mean['week_of_year'] = []
        d_weekly_mean['mean'] = []
        d_weekly_mean['days_in_week'] = []
        d_weekly_mean['first_day_in_week'] = []
        d_weekly_mean['last_day_in_week'] = []

        for week in df['week_of_year'].unique():
            df_week = df[df['week_of_year'] == week]
            d_weekly_mean['week_of_year'].append(week)
            d_weekly_mean['mean'].append(np.mean(df_week[f'diff_{self._ticker_name}']))
            d_weekly_mean['days_in_week'].append(df_week.shape[0])
            d_weekly_mean['first_day_in_week'].append(min(df_week['original_date']))
            d_weekly_mean['last_day_in_week'].append(max(df_week['original_date']))

        d = pd.DataFrame(d_weekly_mean)
        d = d.set_index('week_of_year', drop=True)
        d = d[d['days_in_week'] >= threshold_days_in_week]

        return d

    def weekly_correlation(self, df_weekly_mean, tickers_list, min_prcnt_data, threshold=1):

        main_df_weekly_mean = df_weekly_mean[['mean']]
        main_df_weekly_mean = main_df_weekly_mean.drop(main_df_weekly_mean.index[0])

        main_df_weekly_mean.index = self.reindex(main_df_weekly_mean.index)
        main_df_weekly_mean.columns = ['main_df_weekly_mean']

        d = {}
        d['corr'] = []
        d['main_ticker'] = []
        d['not_main_ticker'] = []
        low_week_sampels_dict = {}
        co = 0

        for tick in tickers_list:
            co += 1

            try:

                tick = tick + '.TA'
                dd = featureEng(tick, years_back=self._years_back)
                if (dd._df.shape[0] > self._df.shape[0] * min_prcnt_data) and (
                        max(dd._df.index) >= max(self._df.index)):
                    df = get_data(tick, data_from_csv=True, path_from='weekly_diff', set_as_index='week_of_year',
                                  index_type='object')
                    if df['days_in_week'][-1] >= threshold:
                        main_df_weekly_mean.loc[:, f'weekly_diff'] = df['mean'][:-1]
                        corr = main_df_weekly_mean.corr(method='pearson')['main_df_weekly_mean'][1]
                        d['corr'].append(corr)
                        d['main_ticker'].append(self._ticker_name)
                        d['not_main_ticker'].append(tick)
                    else:
                        low_week_sampels_dict[tick] = df['days_in_week'][-1]

                else:
                    if not dd._df.shape[0] > self._df.shape[0] * min_prcnt_data:
                        reason = f"""main_ticker_row_num = {self._df.shape[0]},\nnot_main_ticker_row_num = {
                        dd._df.shape[0]}\nwhich is less then {min_prcnt_data * 100}%"""

                    elif not max(dd._df.index) >= max(self._df.index):
                        reason = f"""main_ticker_last_date= {max(self._df.index)},\nnot_main_ticker_last_date= {
                        max(dd._df.index)}"""

                #  print(f'\n{"-" * 20}\n{tick} removed\nreason = {reason}\n{"-" * 20}\n')

            except:
                tick

        return pd.DataFrame(d), low_week_sampels_dict

    def reg_df(self, df_weekly, df_corr, threshold=0.3):
        inv_cols = ['first_day_in_week', 'last_day_in_week']
        df_reg = copy.copy(df_weekly[['mean'] + inv_cols][:])
        last_ind = df_weekly.index[-1]
        df_reg = df_reg.drop(df_reg.index[0])
        df_reg.index = self.reindex(df_reg.index)
        df_reg.loc[last_ind, :] = None
        df_reg.columns = [f'{self._ticker_name}_main_symb_weekly_mean_diff'] + inv_cols
        df_corr_filtered = df_corr[abs(df_corr['corr']) >= threshold]
        co = 0
        for tick, corr in zip(df_corr_filtered['not_main_ticker'], df_corr_filtered['corr']):
            co += 1
            # print('weekly_correlation', df_corr_filtered.shape[0] - co, f'{tick} corr = {corr}')
            try:
                tick = tick
                dd = featureEng(ticker_name=tick, years_back=self._years_back)
                df = dd.daily_diff()
                current = f'{tick}_weekly_mean_diff_{str(round(corr, 2))}'
                df_reg.loc[:, current] = dd.weekly_mean(df)['mean']
                nan_count = df_reg.loc[:, current].isna().sum()
                if nan_count >= 1:
                    # remove weeks with nones
                    df_reg.drop(*[current], axis=1, inplace=True)
                    # print(f'{current} contain nans {nan_count}')
            except:
                tick

        return df_reg

    def index_to_year_week(self, dfindx):
        w = -1
        y = -1
        year_week_indx = []
        for h, (i, j) in enumerate(zip(dfindx.year, dfindx.week)):
            if w == 52 and j == 1 and (i - y == 0):
                year_week_indx.append(f'{i + 1}_{j}')
                y = i
            else:
                y = i
                w = j
                year_week_indx.append(f'{i}_{j}')

        return year_week_indx

    def reindex(self, dfinx):
        new_index = []

        for e in dfinx:
            if ('_1' in e) and len(e) == 6:
                new_date = f'{int(e[:4]) - 1}_52'
                new_index.append(new_date)
            else:
                new_index.append(f'{e[:5]}{int(e[5:]) - 1}')

        return new_index

    def _nan_indexs_search(self, df):
        rows_indx_na_values = df.loc[pd.isna(df["close"]), :].index
        return rows_indx_na_values

    def _nan_remover(self, df, rows_indx_na_values):
        df = df.drop(rows_indx_na_values)
        return df

    def df_reg_int(self, df_reg, corr_inter_threshold,corr_inter_threshold_main):

        df_reg_mout = df_reg[[i for i in df_reg.columns if 'main' not in i]]
        col_name = [i for i in df_reg.columns if 'main' in i]
        a = df_reg[col_name[0]]

        df_reg_int_corrs = df_reg_mout.corr(method='pearson')
        df = df_reg_int_corrs.unstack().sort_values().drop_duplicates().sort_values(kind="quicksort")
        df = df[(abs(df) > corr_inter_threshold) & (df != 1)]
        df_reg_int = copy.copy(df_reg[:])
        for i, cor in zip(df.index, df):

            b = df_reg_int[i[0]].values * df_reg_int[i[1]].values
            corr_main_col = np.corrcoef(a[:-1], b[:-1])

            if abs(corr_main_col[0][1]) > corr_inter_threshold_main:
                df_reg_int[f'{i[0]}_INT_{i[1]}_corr_{round(cor, 2)}'] = b

        return df_reg_int

    def _fix_first_day_of_week(self, df):
        g = []
        for x, y in zip(df.index, df.index.dayofweek):
            if y == 6:
                g.append(x + timedelta(days=1))
            else:
                g.append(x)
        df.index = g
        return df
