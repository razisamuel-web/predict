from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


class Evaluator(object):

    def __init__(self,ticker, reg, X_train, y_train, X_test, y_test):
        self._ticker=ticker
        self._reg = reg
        self._X_train = X_train
        self._X_test = X_test
        self._y_test = y_test
        self._y_train = y_train
        self._y_test_pred = self._reg.predict(self._X_test)
        self._y_train_pred = self._reg.predict(self._X_train)

    def statistical_metrics(self):
        d={}
        d[f'rmse_train'] = np.sqrt(mean_squared_error(self._y_train, self._y_train_pred))
        d[f'r2_train'] = r2_score(self._y_train, self._y_train_pred)

        d[f'rmse_test'] = np.sqrt(mean_squared_error(self._y_test, self._y_test_pred))
        d[f'r2_test'] = r2_score(self._y_test, self._y_test_pred)

        return d
