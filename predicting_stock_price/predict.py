from datetime import datetime, timedelta
import pandas as pd

class Predict(object):

    def __init__(self, reg, target_name, df_reg, date_reference, cols,days_interval):
        self._reg = reg
        self._cols = cols
        self._target_name = target_name
        self._df_reg = df_reg
        self._days_interval= days_interval
        self.start_date = datetime.strptime(str(date_reference), "%Y-%m-%d")
        self._date_str = self._x_sample_extract(start_date=self.start_date, days_interval = self._days_interval)
        self._x_sample_row = pd.DataFrame(self._df_reg.loc[self._date_str, self._cols]).T
        self._x_sample = self._x_sample_row.drop(self._target_name, axis=1)
        self._y_next_diff = round(self._reg.predict(self._x_sample)[0], 5)

    def _x_sample_extract(self, start_date,days_interval):

        '''
        'next_week_diff_pred' caculte the predicted diff for next calendar week of given date
        :param df_reg: pd.DF ready for regression
        :param date_reference: date/str/None , recommended to use today
        :return: int , predicted diff
        '''

        end_date = start_date - timedelta(days=days_interval)
        date_range = [i for i in pd.date_range(start=str(end_date), end=str(start_date))]
        for inx in self._df_reg.index:
            index = inx
            for date in date_range:
                if (date>=datetime.strptime(str(inx[0]), "%Y-%m-%d")) and (datetime.strptime(str(inx[1]), "%Y-%m-%d")):
                    break
            break


        return index



    def _reindex(self, dfinx):
        new_index = []

        for e in dfinx:
            if ('_1' in e) and len(e) == 6:
                new_date = f'{int(e[:4]) - 1}_52'
                new_index.append(new_date)
            else:
                new_index.append(f'{e[:5]}{int(e[5:]) - 1}')

        return new_index

    def _get_given_max_exist_date(self, df, date_reference):
        '''
        "get_given_max_exist_date" looking about if giver date is exist as index of the given data frame. if not, its
        pull the max close date
        :param df: pd.DF, row data frame after cleaning before feature engineering
        :param date_reference: given date
        :return: date
        '''
        indx = df.loc[:, 'close'].index
        date_reference = max(indx[indx <= date_reference])
        return date_reference

    def _get_price(self, df, date_reference):
        '''
        "get_price" pulls price of max of equal or lower date for a given date
        :param df: pd.DF, row data frame after cleaning before feature engineering
        :param date_reference: given date
        :return: float, last "close" price
        '''
        p = df.loc[date_reference, 'close']
        return p

    def next_week_behavior(self, df, date_reference=datetime.today()):
        '''
        "next_week_behavior" calculating all next week metrics
        :param df: pd.DF, row data frame after cleaning before feature engineering
        :param date_reference: given date
        :return: Dictionary
        '''
        date_reference = datetime.strptime(date_reference, '%Y-%m-%d') if type(
            date_reference) == str else date_reference
        exist_date_reference = self._get_given_max_exist_date(df=df, date_reference=date_reference)
        p = self._get_price(df=df, date_reference=exist_date_reference)
        next_week_price = round(p + self._y_next_diff, 4)
        percentage_change = round(self._y_next_diff / p, 4)
        date_reference = datetime.date(date_reference)
        exist_date_reference = datetime.date(exist_date_reference)
        next_week_behavior_dict = {'given_date': [date_reference],
                                   'exist_date_reference': [exist_date_reference],
                                   'last_close_price': [p],
                                   'predicted_diff': [self._y_next_diff],
                                   'next_week_price': [next_week_price],
                                   'percentage_change': [percentage_change]
                                   }

        return next_week_behavior_dict

    def next_week_class(self, clf):
        '''
        next_week_class predict next week class by using classifier model which built in 'train class'
        :param clf: sklearn classifier model
        :return: dictionary of predicted class and probabilities for each class
        in order to change convert the dict to pd.Data_frame use pd.DataFrame.from_dict(dict, orient='index').T
        '''

        probs = clf.predict_proba(self._x_sample)[0]
        predicted_class = clf.predict(self._x_sample)[0]

        d = {'class': predicted_class,
             'False_p': probs[0],
             'True_p': probs[1]}

        return d
