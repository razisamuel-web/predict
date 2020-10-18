from plots import plot_pred_actual
from showing_performenc_function import *
import pandas as pd
import pickle
from feature_engineering import featureEng

tick_name = 'MVNE'
test_size = 0.15
date_reference = '2020-08-01'
random_state_plot = 3
years_back = 1.4

df_reg = pd.read_csv(f'./df_regs_full_loop/df_reg_{tick_name}.csv', index_col=['first_day_in_week', 'last_day_in_week'])
pkl_filename = f'pickel_models/{tick_name}/model.pkl'
with open(pkl_filename, 'rb') as file:
    reg = pickle.load(file)

dfr = get_data(ticker_name=tick_name, data_from_csv=1, path_from='raw_data')
dfc = get_data(ticker_name=tick_name, data_from_csv=1, path_from='data')

daily_diff = featureEng(ticker_name=tick_name,
                        years_back=years_back,
                        data_from_csv=True,
                        path_from='data').daily_diff()

# %%
resid, r2, f, df_reg_summary = generate_df_reg_summary(tick_name, reg=reg, df_reg_columns=df_reg.columns)
print(r2, '\n', f)
df_reg_summary


# %%
result_summary(tick_name, path='./results/2020-08-01.csv')

# %%
get_rolling_mean_std(df_raw_close=dfc[['close']],
                     df_daily_diff=daily_diff,
                     date_reference=date_reference,
                     mean_days_roll=3,
                     std_days_roll=10,
                     show_days_range_back=6)
# %%
plot_pred_actual(df_regression=df_reg,
                 main_ticker=tick_name,
                 test_size=test_size,
                 random_state=random_state_plot,
                 reg=reg)
# %%
clean_raw_and_weekly_mean_plot(tick_name)

# %% plot 5 most correlated by weeks
correlated_ticks_plot(tick_name=tick_name,
                      top=8,
                      weeks_back=[5],
                      df_reg_summary=df_reg_summary)

# %% daily absolute value plot
df = daily_correlated_ticks_plot(tick_name=tick_name,
                                 df_reg_columns=df_reg.columns,
                                 weeks_back=3)
