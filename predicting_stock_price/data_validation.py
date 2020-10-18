import pandas as pd
from datetime import datetime
import numpy as np
from data_generation import get_data, save_as_csv
import pickle
import math
import os


class DataValidation(object):

    def __init__(self, df, column):
        self._column = column
        self._df = df
        self._df_count = len(self._df.index)
        self._is_low_amount_of_data = 1 if self._df_count < 121 else 0
        self._nan_count = self.nan_volume(df)
        self._nan_non_nan_ratio = round(self._nan_count / self._df_count, 6)
        self._nan_sequences_count_and_indexes = self._nan_len_index(df=df, col=column)

        self._is_exist_nan_sequence = 1 if len(self._nan_sequences_count_and_indexes['count']) > 0 else 0
        self._number_of_nan_sequences = len(self._nan_sequences_count_and_indexes['count'])
        self._mean_of_nan_sequences_count = np.mean(
            self._nan_sequences_count_and_indexes['count']) if self._is_exist_nan_sequence == 1 else 0

        self._is_nan_non_nan_ratio_problematic = 1 if self._nan_non_nan_ratio > 0.2 else 0
        self._df_decimal_point_changed = self.is_decimal_point_changed(df)
        self._decimal_point_changed_indexs = self._df_decimal_point_changed.index
        self._length_of_decimal_point_changes = len(self._decimal_point_changed_indexs)
        self._is_decimal_point_changed = True if self._length_of_decimal_point_changes > 1 else False
        self._ticker_days_diff = self.ticker_stop_days_diff(df)
        self._is_ticker_stopped = True if self._ticker_days_diff > 7 else False
        self._data_not_changed_dict = self._data_not_changed(df, column=self._column, threshold=7)
        self._is_data_not_changed = 1 if len(self._data_not_changed_dict['value']) > 0 else 0

    def nan_volume(self, df):
        number_of_nans = sum(df.apply(lambda x: sum(x.isnull().values), axis=1) > 0)
        return number_of_nans

    def is_decimal_point_changed(self, df):
        d = df[:]
        d = d[[self._column]]
        result = d[self._column].rolling(7).mean().dropna()
        d.loc[:, 'weekly_mean'] = result
        d.loc[:, 'weekly_mean_ratio'] = d[self._column] / d['weekly_mean']
        df_decimal_point_changed = d[d['weekly_mean_ratio'] < 0.1]

        return df_decimal_point_changed

    def ticker_stop_days_diff(self, df):
        duration = datetime.now() - max(df.index)  # For build-in functions
        duration_in_s = duration.total_seconds()
        days_diff = divmod(duration_in_s, 86400)[0]
        return days_diff

    def _data_not_changed(self, df, column, threshold):
        d = {}
        c = 0
        d['value'] = []
        d['count'] = []
        d['indexes'] = []
        j = df[column][0:1][0]
        for i, indx in zip(df[column][1:], df[column][1:].index):
            if j == i:
                end_indx = indx
                c += 1
                if c == 1:
                    start_indx = indx
            elif c >= threshold:
                c += 1
                d['value'].append(j)
                d['count'].append(c)
                d['indexes'].append((start_indx, end_indx))
                c = 0
                end_indx = indx

            else:
                c = 0

            j = i
        return d

    def _nan_len_index(self, df, col):
        '''
        "nan_len_index is indexing the nan sequences, and count the number of each sequence
        :param df: pandas data frame
        :param col: string
        :return: dictionary of nan indexes and the len of each sequence
        '''
        c = 0
        d = {}
        d['index'] = []
        d['count'] = []
        start_index = ii = df.index[0]
        for i, j in zip(df.index, df[col]):
            start_index = ii if c == 0 else start_index
            if math.isnan(j):
                c += 1
            else:
                if c > 0:
                    d['index'].append((start_index, i))
                    d['count'].append(c)

                c = 0

            ii = i
        return d


# TODO : create class variable as self._tickers_list_of_decimal_point_chamged
# TODO : adding function which fixed decimal point isuues in case of recurrent corruptions


class DataValidationLoop(object):

    def __init__(self, tickers_list, column, data_from_csv=True, path_from='data'):
        self._column = column
        self._tickers_list = tickers_list
        self._data_from_csv = data_from_csv
        self._path_from = path_from
        self._dvl_df = self.data_validation_all_tickers()

    def data_validation_all_tickers(self):
        dv_dict = {}
        dv_dict['is_low_amount_of_data'] = []
        dv_dict['df_count'] = []
        dv_dict['ticker_name'] = []
        dv_dict['is_nan_non_nan_ratio_problematic'] = []
        dv_dict['nan_count'] = []
        dv_dict['nan_non_nan_ratio'] = []

        dv_dict['is_exist_nan_sequence'] = []
        dv_dict['nan_sequences_count_and_indexes'] = []

        dv_dict['number_of_nan_sequences'] = []
        dv_dict['mean_of_nan_sequences_count'] = []

        dv_dict['nan_sequences_count_and_indexes'] = []

        dv_dict['is_decimal_point_changed'] = []
        dv_dict['number_of_corrupted_decimal_point'] = []
        dv_dict['decimal_point_changed_indexs'] = []
        dv_dict['is_ticker_stopped'] = []
        dv_dict['ticker_stopped_diff'] = []
        dv_dict['is_ticker_data_not_changed'] = []
        dv_dict['ticker_data_not_changed_dictionary'] = []
        dv_dict['is_ticker_data_not_exist'] = []

        c = 0

        for ticker in self._tickers_list:
            c += 1
            print('DVL', len(self._tickers_list) - c)
            df_ticker = get_data(ticker, self._data_from_csv, path_from=self._path_from)
            if isinstance(df_ticker, pd.DataFrame):

                DV = DataValidation(df_ticker, column=self._column)
                dv_dict['ticker_name'].append(ticker)
                dv_dict['is_low_amount_of_data'].append(DV._is_low_amount_of_data)
                dv_dict['df_count'].append(DV._df_count)
                dv_dict['is_nan_non_nan_ratio_problematic'].append(DV._is_nan_non_nan_ratio_problematic)
                dv_dict['nan_count'].append(DV._nan_count)
                dv_dict['nan_non_nan_ratio'].append(DV._nan_non_nan_ratio)

                dv_dict['is_exist_nan_sequence'].append(DV._is_exist_nan_sequence)
                dv_dict['nan_sequences_count_and_indexes'].append(DV._nan_sequences_count_and_indexes)

                dv_dict['number_of_nan_sequences'].append(DV._number_of_nan_sequences)
                dv_dict['mean_of_nan_sequences_count'].append(DV._mean_of_nan_sequences_count)

                dv_dict['is_decimal_point_changed'].append(DV._is_decimal_point_changed)
                dv_dict['is_ticker_stopped'].append(DV._is_ticker_stopped)
                dv_dict['number_of_corrupted_decimal_point'].append(DV._length_of_decimal_point_changes)
                dv_dict['decimal_point_changed_indexs'].append(DV._decimal_point_changed_indexs)
                dv_dict['ticker_stopped_diff'].append(DV._ticker_days_diff)
                dv_dict['is_ticker_data_not_changed'].append(DV._is_data_not_changed)
                dv_dict['ticker_data_not_changed_dictionary'].append(DV._data_not_changed_dict)
                dv_dict['is_ticker_data_not_exist'].append(0)
            else:
                dv_dict['ticker_name'].append(ticker)
                dv_dict['is_ticker_data_not_exist'].append(1)
                for key in dv_dict.keys():
                    if (key != 'ticker_name') and (key != 'is_ticker_data_not_exist'):
                        dv_dict[key].append(None)

        return pd.DataFrame(dv_dict)

    def remove_problematic_tickers_from_ticker_list(self,
                                                    df,
                                                    by_is_low_amount_of_data=True,
                                                    by_is_nan_non_nan_ratio_problematic=True,
                                                    by_is_decimal_point_changed=True,
                                                    by_is_ticker_stopped=True,
                                                    by_is_ticker_data_not_exist=True,
                                                    by_is_ticker_data_not_changed=True
                                                    ):

        by_is_low_amount_of_data = by_is_low_amount_of_data if by_is_low_amount_of_data == True else None
        df = df[df['is_low_amount_of_data'] != by_is_low_amount_of_data]
        print(df.shape, '  is_low_amount_of_data')

        by_is_nan_non_nan_ratio_problematic = by_is_nan_non_nan_ratio_problematic if by_is_nan_non_nan_ratio_problematic == True else None
        df = df[df['is_nan_non_nan_ratio_problematic'] != by_is_nan_non_nan_ratio_problematic]
        print(df.shape, '  is_nan_non_nan_ratio_problematic')

        by_is_decimal_point_changed = by_is_decimal_point_changed if by_is_decimal_point_changed == True else None
        df = df[df['is_decimal_point_changed'] != by_is_decimal_point_changed]
        print(df.shape, '  is_decimal_point_changed')

        by_is_ticker_stopped = by_is_ticker_stopped if by_is_ticker_stopped == True else None
        df = df[df['is_ticker_stopped'] != by_is_ticker_stopped]
        print(df.shape, '  is_ticker_stopped')

        by_is_ticker_data_not_exist = by_is_ticker_data_not_exist if by_is_ticker_data_not_exist == True else None
        df = df[df['is_ticker_data_not_exist'] != by_is_ticker_data_not_exist]
        print(df.shape, '  is_ticker_data_not_exist')

        # by_is_ticker_data_not_changed = by_is_ticker_data_not_changed if by_is_ticker_data_not_changed == True else None
        # df = df[df['is_ticker_data_not_changed'] != by_is_ticker_data_not_changed]
        # print(df.shape, '  is_ticker_data_not_changed')

        return list(df['ticker_name'])

    def invstigation_of_specific_ticker(self, ticker_name):
        d = get_data(ticker_name, data_from_csv=True)
        dv = DataValidation(d)
        return dv

    def _decimal_point_recovery(self, df, column, change_threshold=0.4):
        start_n_max = max(df[column])
        start_index = df[df[column] == start_n_max].index[0]
        indexs_down_way = [inx for inx in df.index if inx > start_index]
        indexs_up_way = [inx for inx in df.index[::-1] if inx < start_index]
        indx_up_down = [indexs_down_way, indexs_up_way]
        df_fixing = df[[column]][:]

        for j in indx_up_down:
            start_n = max(df[column])

            for inx_dwn in j:
                n = df_fixing.loc[inx_dwn, :][0]

                if not np.isnan(np.float(n)):
                    ratio = n / start_n
                    last = inx_dwn

                    if ratio < change_threshold:
                        ## TODO : if ratio != 1/10/100 , some other curaaption detection I(nor only decimal corruption)
                        a = len(str(int(start_n)))
                        b = len(str(int(n))) if n >= 1 else self._decimal_frection(n)
                        n = n * (10 ** (a - b))
                        n = n / 10 if n / start_n > 3 else n
                        n = n * 10 if n / start_n < 0.2 else n
                        df_fixing.loc[inx_dwn, column] = n

                    start_n = n

        df.loc[:, f'{column}_old'] = list(df[column][:])
        df[column] = df_fixing[column]

        return df

    def _decimal_frection(self, number):
        c = -1
        while number < 1: number *= 10;c += 1
        return c

    def _remove_not_changed_data(self, df, threshold, count_indxs_dict):
        '''

        :param df: pandas dataFrame
        :param threshold: min days to declare on not changigng data
        :param count_indxs_dict: indexes of problematic data , count of each range
        :return: clean data frame, pay attention, each first day of problematic data remain and last day remove

        '''
        for c, indx in zip(count_indxs_dict['count'], count_indxs_dict['indexes']):
            if c > threshold:
                start_date = indx[0]
                end_date = indx[1]
                mask = (df.index < start_date) | (df.index > end_date)
                df = df.loc[mask]

        return df

    def _nan_len_recovery(self, col, df, df_nan, threshold=4):
        '''
        "nan_len_recovery" fixing nan sequences  by spanning with respect to the values of the two limits
        :param col: string
        :param df: pandas data frame
        :param df_nan: pd.DF OF nan indexing and len of sequences
        :return:
        '''
        df_nan = pd.DataFrame(df_nan)
        df_nan_clean = df_nan[df_nan['count'] <= threshold]
        df.index = pd.to_datetime(df.index)

        for i, c in zip(df_nan_clean['index'], df_nan_clean['count']):
            l = np.linspace(df.loc[i[0], col], df.loc[i[1], col], c + 2)
            df.loc[i[0]: i[1], col] = l
        return df

    def _remove_not_changed_data_and_save_tickers(self, tickers_list_not_changed_data):
        c = 0
        l = tickers_list_not_changed_data
        for ticker in l:
            c += 1
            print('REMOVE NOT CHANGED DATA', ticker, len(l) - c)
            count_indxs_dict = \
                self._dvl_df[self._dvl_df.ticker_name == ticker].ticker_data_not_changed_dictionary.values[0]
            df_ticker = get_data(ticker, data_from_csv=1, path_from='data')
            df_ticker = self._remove_not_changed_data(df_ticker, threshold=4, count_indxs_dict=count_indxs_dict)
            save_as_csv(ticker, df_ticker, outdir='./data')

    def _decimal_point_recovery_and_save_tickers(self, tickers_list_not_decimal_point_changed, column,
                                                 change_threshold=0.4):
        l = tickers_list_not_decimal_point_changed
        c = 0
        for ticker in l:
            c += 1
            print('decimal point recovery', ticker, len(l) - c)
            df_ticker = get_data(ticker, data_from_csv=1, path_from='data')
            df_ticker = self._decimal_point_recovery(df_ticker, column=column, change_threshold=change_threshold)
            save_as_csv(ticker, df_ticker, outdir='./data')

    def _nan_len_recovery_and_save_tickers(self, tickers_list_with_nan_sequences, column, threshold=4):

        # nan_len_recovery(self, col, df, df_nan, threshold=4)

        l = tickers_list_with_nan_sequences
        c = 0
        for ticker in l:
            c += 1
            print('nan_sequence_recovery', ticker, len(l) - c)
            indexes_count_dict = \
                self._dvl_df[self._dvl_df.ticker_name == ticker].nan_sequences_count_and_indexes.values[0]
            df_ticker = get_data(ticker, data_from_csv=1, path_from='data')
            df_ticker = self._nan_len_recovery(col=column, df=df_ticker, df_nan=indexes_count_dict, threshold=threshold)
            save_as_csv(ticker, df_ticker, outdir='./data')


class DataValidationCorrection(object):

    def __init__(self, tickers, columns: list=['open', 'close']):
        self._columns = columns
        self._clean_ticker_list = tickers
        for i in self._columns:
            print(f'========   {i}   ========')
            self._clean_ticker_list, self._df_problematic = self.correction(i, self._clean_ticker_list)
            if not os.path.exists('df_problematic'):
                os.mkdir('df_problematic')
            self._df_problematic.to_csv(f'./df_problematic/df_problematic_{i}.csv')

        # self._first_clean_ticker_list = self.correction(self._columns[0], self._tickers)
        # self._clean_ticker_list = self.correction(self._columns[1], self._first_clean_ticker_list)

        with open("./symbols/clean_ticker_list.txt", "wb") as fp:  # Pickling
            pickle.dump(self._clean_ticker_list, fp)
            print('clean ticker list saved')

    def correction(self, col, tickers, nan_seq_recov_thresh=4):
        DVL = DataValidationLoop(tickers_list=tickers, column=col, data_from_csv=True, path_from='data')

        df_problematic = DVL._dvl_df

        l = df_problematic[df_problematic.is_decimal_point_changed == 1]['ticker_name']
        l_decimal_point_changed = list(l)
        DVL._decimal_point_recovery_and_save_tickers(l_decimal_point_changed, column=col)

        l = df_problematic[df_problematic.is_exist_nan_sequence == 1]['ticker_name']
        l_nan_sequences = list(l)
        DVL._nan_len_recovery_and_save_tickers(l_nan_sequences, column=col, threshold=nan_seq_recov_thresh)

        # remove not changed price
        # l = df_problematic[df_problematic.is_ticker_data_not_changed == 1]['ticker_name']
        # l_no_data_change = list(l)
        # DVL._remove_not_changed_data_and_save_tickers(l_no_data_change)

        DVL = DataValidationLoop(tickers_list=tickers, column=col, data_from_csv=True, path_from='data')
        df_problematic = DVL._dvl_df

        clean_ticker_list = DVL.remove_problematic_tickers_from_ticker_list(df_problematic,
                                                                            by_is_low_amount_of_data=True,
                                                                            by_is_nan_non_nan_ratio_problematic=True,
                                                                            by_is_decimal_point_changed=True,
                                                                            by_is_ticker_data_not_exist=True,
                                                                           # by_is_ticker_data_not_changed=True
                                                                            )
        return clean_ticker_list, df_problematic
