import matplotlib.pyplot as plt
import pandas as pd
from train import Train,ClfTrain
from evaluator import Evaluator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


def plot_pred_actual(df_regression,
                     main_ticker,
                     test_size,
                     random_state,
                     reg ):

    df = df_regression[:]

    X_train, y_train, X_test, y_test = ClfTrain(main_ticker)._generate_x_y(tick_name=main_ticker)

    train_indx, test_index = train_test_split(df.index, test_size=test_size, random_state=random_state)

    y_train_pred =  reg.predict(X_train)
    y_test_pred =  reg.predict(X_test)

    df_summary = pd.DataFrame({'y_train': y_train, 'y_train_pred': y_train_pred}, index=train_indx)
    df_summary.sort_index(axis=0, inplace=True)
    df_summary.index = [f'{i[0][2:]}\n{i[1][5:]}' for i in df_summary.index]
    df_summary.plot(style='-o')
    plt.show()

    df_summary = pd.DataFrame({'y_train': y_test, 'y_train_pred': y_test_pred}, index=test_index)
    df_summary.sort_index(axis=0, inplace=True)
    df_summary.index = [f'{i[0][2:]}\n{i[1][5:]}' for i in df_summary.index]
    df_summary.plot(style='-o')
    plt.show()



def adding_date_to_week_of_year(df):
    new_indx = []
    for i in df.index:
        ci = i
        ci = datetime.strptime(ci.replace('_', '-W') + '-1', '%G-W%V-%u')

        cis = ci + timedelta(days=7)
        cis = cis.strftime('%m-%d')
        cis = str(cis)

        cie = ci + timedelta(days=13)
        cie = cie.strftime('%m-%d')
        cie = str(cie)

        new_indx.append(i + '\n' + cis + '\n' + cie)

    df.index = new_indx

    df = df.reset_index()

    return df


def fix_year_num(s):
    for i in range(1, 10):
        if f'_{i}\n' in s:
            s = s.replace(f'_{i}\n', f'_0{i}\n')
    return s


def fix_year_num_of_df(df):
    inx = [fix_year_num(i) for i in df.index]
    df.index = inx
    return df

def sort_data_frame(df):
    df['indexNumber'] = df.index.str.slice(0, 8).astype(int)
    df = df.sort_values(['indexNumber']).drop('indexNumber', axis=1)
    return df
