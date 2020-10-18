from yahoo_fin import stock_info as si
from datetime import date, timedelta, datetime
import pandas as pd
import os


# TODO: some ticker names are not pronounsed correct

class DataGeneration(object):

    def __init__(self, tickers_list, start_date, years_back_data_generation=1, path='raw_data'):
        self.tickers_list = tickers_list
        self._years_back = years_back_data_generation
        self._start_date_str = str(date.today())
        self._start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self._same_date_last_year = str(self._start_date - timedelta(days=round(years_back_data_generation * 365)))
        self._path = path

    def download_to_csv(self):
        for ticker in self.tickers_list:
            try:
                ticker = ticker + '.TA' if '.TA' not in ticker else ticker
                ticker = ticker.replace('.L.', '-L.')
                df = si.get_data(ticker=ticker,
                                 start_date=self._same_date_last_year,
                                 end_date=self._start_date_str,
                                 index_as_date=True,
                                 interval="1d")

                save_as_csv(ticker, df, outdir=f'./{self._path}')
            except:
                print(f'ticker name = {ticker} not exist')

    def duplicate_date(self, path_from='raw_data', path_to='data'):
        c = 0
        for ticker in self.tickers_list:
            c += 1
            print(len(self.tickers_list) - c)
            try:
                df = get_data(ticker_name=ticker, data_from_csv=1, path_from=path_from)
                save_as_csv(ticker=ticker, df=df, outdir=f'./{path_to}')
            except:
                print(f'ticker name = {ticker} not exist')


def get_data(ticker_name, data_from_csv, path_from='data', set_as_index='Unnamed: 0', index_type='datetime64[ns]'):
    if data_from_csv:
        try:
            ticker_name = ticker_name + '.TA' if 'TA' not in ticker_name else ticker_name
            ticker_name = ticker_name.replace('.L.', '-L.')
            df = pd.read_csv(f'./{path_from}/{ticker_name}.csv')
            df = df.set_index(set_as_index, drop=True)
            df.index = df.index.astype(index_type)
            return df

        except:
            print(f'make sure {ticker_name} is exist downloaded and convert to csv')


def save_as_csv(ticker, df, outdir='./data'):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    ticker = ticker + '.TA' if '.TA' not in ticker else ticker
    ticker += '.csv'
    fullname = os.path.join(outdir, ticker)
    df.to_csv(f'{fullname}')
    #print(f'ticker name = {ticker} saved as {fullname}')
