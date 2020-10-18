import pandas as pd
import pickle
from general_functions import weekly_correlations_to_csv
from data_validation import DataValidationCorrection
from data_generation import DataGeneration
import json

configs = json.loads(open('configs.json', 'r').read())

NAME = configs["NAME"]
random_state_plot = configs["random_state_plot"]
test_size = configs["test_size"]
years_back = configs["years_back"]
threshold_days_in_week = configs["threshold_days_in_week"]
min_percentage_data = configs["min_percentage_data"]
min_days_in_week = configs["min_days_in_week"]
corr_threshold = configs["corr_threshold"]
corr_inter_threshold = configs["corr_inter_threshold"]
days_interval = configs["days_interval"]
models_path = configs["models_path"]
corr_inter_threshold_main = configs["corr_inter_threshold_main"]
date_reference_data_generation = configs["date_reference_data_generation"]
date_reference_end = configs["date_reference_end"]
correlation_path = configs["correlation_path"]
years_back_data_generation = configs["years_back_data_generation"]
tickers = list(pd.read_csv('./symbols/symbols.csv')['symbol'])

# %%
datageneration = DataGeneration(tickers_list=tickers,
                                years_back_data_generation=years_back_data_generation,
                                start_date=date_reference_data_generation,
                                path='raw_data')

# datageneration.download_to_csv()
# %% In order to run full process u need only to duplicate the data and rerun
datageneration.duplicate_date(path_from='raw_data', path_to='data')

# %%

DVC = DataValidationCorrection(tickers=tickers, columns=['close'])

# %%
with open("./symbols/clean_ticker_list.txt", "rb") as fp:  # Unpickling
    clean_ticker_list = pickle.load(fp)
    print(f'number of ticker in the beegining {len(tickers)}')
    print(f'number of ticker after validation {len(clean_ticker_list)}')
    print(f'diff is = {len(tickers) - len(clean_ticker_list)}')

# %%
weekly_correlations_to_csv(tickers_list=clean_ticker_list,
                           years_back_data_generation=years_back_data_generation,
                           start_date=date_reference_data_generation,
                           days_interval=days_interval,
                           threshold_days_in_week=threshold_days_in_week,
                           path=correlation_path)

