from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pickle
import os
from os.path import isfile, join
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from dev_reg_summary import summary


class Trainn(object):

    def __init__(self, ticker_name, df_reg, test_size, path):
        self.model = linear_model.Ridge
        self._ticker_name = ticker_name
        self._outdir = f'{path}/{ticker_name}'
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(self._outdir):
            os.mkdir(self._outdir)

        self._test_size = test_size
        self._target_column_name = f'{self._ticker_name}_main_symb_weekly_mean_diff'
        self.df_reg = df_reg
        self.df_corrs_summary = self.train_corrs_cv(df_reg=self.df_reg,
                                                    test_size=self._test_size)

        self.df_alphas_filtered = self.df_corrs_summary[self.df_corrs_summary['alpha'] <= 1]
        self.df_alphas_filtered = self.df_corrs_summary if self.df_corrs_summary.shape[0] > 0 else \
            self.df_corrs_summary[self.df_corrs_summary['alpha'] == min(self.df_corrs_summary['alpha'])]

        self.df_filtered = self.df_alphas_filtered[
            self.df_alphas_filtered['rmse_test'] == min(self.df_alphas_filtered['rmse_test'])]
        self.df_filtered_dict = self.df_filtered.to_dict('r')[0]
        self.reg = self._remove_not_chosen_models()

        self._f_path = './df_trained_filtered_dict'
        if not os.path.exists(self._f_path):
            os.mkdir(self._f_path)

        a_file = open(f"{self._f_path}/{self._ticker_name}.pkl", "wb")
        pickle.dump(self.df_filtered_dict, a_file)
        a_file.close()

    def _remove_not_chosen_models(self):
        self._files_list = [f for f in os.listdir(self._outdir) if isfile(join(self._outdir, f))]
        self._files_list = [i for i in self._files_list if i not in self.df_filtered_dict['reg']]
        for f in self._files_list:
            current_path = f'{self._outdir}'
            os.remove(os.path.join(current_path, f))
        os.rename(self.df_filtered_dict['reg'], f'{self._outdir}/model.pkl')
        self.df_filtered_dict['reg'] = f'{self._outdir}/model.pkl'
        p = f'{self._outdir}/model.pkl'
        with open(p, 'rb') as file:
            reg = pickle.load(file)
        return reg

    def split_to_train_test(self, df_reg, test_size=0.15, random_state=3):
        y = df_reg[self._target_column_name].values
        X = df_reg.drop([self._target_column_name], axis=1).values
        split = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return split

    def fittrain(self, X_train, y_train, alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3]):
        reg = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
        reg.fit(X_train, y_train)

        return reg

    def fittrainGsCv(self, X_train, y_train):
        steps = [
            ('scalar', StandardScaler()),
            #  ('poly', PolynomialFeatures(degree=1)),
            ('model', self.model())
        ]

        ridge_pipe = Pipeline(steps)

        scoring_func = make_scorer(mean_squared_error)

        parameters = [{'model__alpha': [1e-4,
                                        1e-3,
                                        1e-2,
                                        1e-1,
                                        1,
                                        1e+1,
                                        # 1e+2
                                        ],
                       'model__tol': [0.0001,
                                      0.001,
                                      # 0.01
                                      ],
                       'model__max_iter': [50,
                                           # 70,
                                           # 90
                                           ],
                       'model__fit_intercept': [True],
                       'model__normalize': [True]}
                      ]

        grid_search = GridSearchCV(estimator=ridge_pipe,
                                   param_grid=parameters,
                                   scoring=scoring_func,
                                   cv=3,
                                   n_jobs=-1)

        grid_search = grid_search.fit(X_train, y_train)

        grid_search.alpha_ = grid_search.best_params_['model__alpha']
        grid_search.coef_ = grid_search.best_estimator_._final_estimator.coef_[
                            :]  # .coef_[1:] in the past the first coef was all the time 0 in some point its changed
        grid_search.intercept_ = grid_search.best_estimator_._final_estimator.intercept_

        return grid_search

    def train_corrs_cv(self, df_reg, test_size):
        from general_functions import corrs_mean_from_cols_names

        df_reg_test = df_reg
        columns = df_reg_test.columns
        treget_column_name = columns[0]

        corrs, corrs_mean = corrs_mean_from_cols_names(columns)

        corrs_str = columns[1:]

        X = corrs_str
        Y = corrs
        corrs_str = [x for _, x in sorted(zip(Y, X))]
        corrs = sorted(corrs)

        r2s_train = []
        r2_ad_nps_train = []
        rmses_train = []

        r2s_test = []
        r2_ad_nps_test = []
        rmses_test = []

        alphas = []
        corr_means = []
        current_corrs_strs = []

        regs = []

        predictors_num = []

        x_trains = []
        x_tests = []
        y_trains = []
        y_train_preds = []
        y_tests = []
        y_test_preds = []

        model_name = 0
        for c in range(0, len(corrs_str)):
            if c == 0:  # order by correlation
                current_corrs_str = [treget_column_name] + corrs_str[c:]
                current_df_reg = df_reg_test[current_corrs_str][:]

            corra_mean = np.mean(corrs[c:])

            X_train, X_test, y_train, y_test = self.split_to_train_test(current_df_reg,
                                                                        test_size=test_size,
                                                                        random_state=3)
            # reg = self.fittrain(X_train=X_train, y_train=y_train)
            reg = self.fittrainGsCv(X_train=X_train, y_train=y_train)

            # order by p_value
            p_summary = summary(clf=reg, X=X_train, y=y_train, xlabels=current_df_reg.columns[1:])
            current_corrs_str = p_summary.iloc[1:, :].sort_values(by='p_value', ascending=False).index.insert(0,
                                                                                                              treget_column_name)


            current_df_reg = df_reg_test[current_corrs_str][:]
            # print('\n======       ', c, '   =====')

            y_train_pred = reg.predict(X_train)
            y_test_pred = reg.predict(X_test)

            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)

            p = len(current_corrs_str) - 1

            n_train = len(y_train) + p
            r2_ad_np_train = 1 - (1 - r2_train) * ((n_train - 1) / (n_train - p))

            n_test = len(y_test) + p
            r2_ad_np_test = 1 - (1 - r2_test) * ((n_test - 1) / (n_test - p))

            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

            alpha = reg.alpha_

            x_trains.append(X_train)
            x_tests.append(X_test)
            y_trains.append(y_train)
            y_train_preds.append(y_train_pred)
            y_tests.append(y_test)
            y_test_preds.append(y_test_pred)

            current_corrs_strs.append(current_corrs_str)
            corr_means.append(corra_mean)

            r2s_train.append(r2_train)
            r2s_test.append(r2_test)

            r2_ad_nps_train.append(r2_ad_np_train)
            r2_ad_nps_test.append(r2_ad_np_test)

            rmses_train.append(rmse_train)
            rmses_test.append(rmse_test)

            alphas.append(alpha)

            model_name += 1
            pkl_filename = f"{self._outdir}/pickle_model_{model_name}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(reg, file)

            regs.append(pkl_filename)

            predictors_num.append(p)

        dict = {
            'x_train': x_trains,
            'x_test': x_tests,
            'y_train': y_trains,
            'y_train_pred': y_train_preds,
            'y_test': y_tests,
            'y_test_pred': y_test_preds,

            'current_corrs_str': current_corrs_strs,
            'r2_train': r2s_train,
            'r2_test': r2s_test,

            'r2_ad_nps_train': r2_ad_nps_train,
            'r2_ad_nps_test': r2_ad_nps_test,

            'rmse_train': rmses_train,
            'rmse_test': rmses_test,

            'alpha': alphas,
            'corra_mean': corr_means,
            'predictor_num': predictors_num,
            'reg': regs,
        }

        df_corrs_summary = pd.DataFrame(dict)

        return df_corrs_summary


import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


class ClfTrain(object):

    def __init__(self, tick_name):
        self._tick_name = tick_name
        self._X_train, self._y_train_cont, self._X_test, self._y_test_cont = self._generate_x_y(self._tick_name)
        self._y_train = self._y_train_cont > 0
        self._y_test = self._y_test_cont > 0
        self._pipeline = self._pipelinedef()
        self._n_comp = int(
            round(self._X_train.shape[0] / 2, 0) if self._X_train.shape[0] / 2 < self._X_train.shape[1] else
            self._X_train.shape[1])
        self._lr_n_components = list(range(1, self._n_comp + 1, 1))
        self._lr_C = np.logspace(-4, 4, 50)
        self._lr_penalty = ['l2', 'l1']
        self._parameters = dict(pca__n_components=self._lr_n_components,
                                logistic__C=self._lr_C,
                                logistic__penalty=self._lr_penalty)

        self._lr_clf = GridSearchCV(self._pipeline, self._parameters)

    def _generate_x_y(self, tick_name):
        a_file = open(f"./df_trained_filtered_dict/{tick_name}.pkl", "rb")
        output = pickle.load(a_file)
        return output['x_train'], output['y_train'], output['x_test'], output['y_test']

    def _pipelinedef(self, params=False):
        if not params:
            sc = StandardScaler()
            pca = decomposition.PCA()
            logistic = linear_model.LogisticRegression()

            steps = Pipeline(steps=[('sc', sc),
                                    ('pca', pca),
                                    ('logistic', logistic)])
            return steps

        else:
            sc = StandardScaler() if not params['sc'] else StandardScaler(**params['sc'])
            pca = decomposition.PCA() if not params['pca'] else decomposition.PCA(**params['pca'])
            logistic = linear_model.LogisticRegression() if not params['logistic'] else linear_model.LogisticRegression(
                **params['logistic'])

            steps = Pipeline(steps=[('sc', sc),
                                    ('pca', pca),
                                    ('logistic', logistic)])

            return steps

    def fit_lr_gridsearch_cv(self, save_model_path='./pickel_lr_models'):
        clf = self._lr_clf.fit(self._X_train, self._y_train)

        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)

        pkl_filename = f"{save_model_path}/pickle_lr_model_{self._tick_name}.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)

        return clf

    def generate_lr_summary(self, clf):
        lr_summary_dict = {
            'Best_Penalty:': clf.best_estimator_.get_params()['logistic__penalty'],
            'Best_C:': clf.best_estimator_.get_params()['logistic__C'],
            'logistic:': clf.best_estimator_.get_params()['logistic']
        }

        return lr_summary_dict

    def generate_clf_partial_summary(self, clf):
        y_pred = clf.predict(self._X_test)
        class_summary_dict = {
            'confusion_matrix_test:': confusion_matrix(self._y_test, y_pred),
            'classification_report': classification_report(self._y_test, y_pred),
            'accuracy_test': sum(i == j for i, j in zip(y_pred, self._y_test)) / len(y_pred)
        }

        return class_summary_dict

    def generate_clf_summary(self, clf, classifire_type='lr'):
        class_summary_dict = self.generate_lr_summary(clf)

        if classifire_type == 'lr':
            current_class_type_dict = self.generate_clf_partial_summary(clf)

        class_summary_dict.update(current_class_type_dict)

        return class_summary_dict

    def predict_actual_diffs(self, clf):
        probs = pd.DataFrame(np.round(clf.predict_proba(self._X_test), 2), columns=['0', '1'])
        probs['pred'] = clf.predict(self._X_test)
        probs['test'] = self._y_test
        probs['diff'] = round(abs(probs['0'] - probs['1']), 2)
        probs['y_contin_norm'] = np.abs(np.round((self._y_test_cont - np.mean(self._y_test_cont)) /
                                                 np.std(self._y_test_cont), 2))
        probs['y_contin'] = np.round(self._y_test_cont, 2)

        probs['is_correct'] = probs['pred'] == probs['test']
        return probs

    def learn_with_chosen_params(self, cls_cv_best_params, x, y):
        logistic_best_param_dict = {i.replace('logistic__', ''): j for i, j in cls_cv_best_params.items() if
                                    'logistic' in i}
        pca_best_param_dict = {i.replace('pca__', ''): j for i, j in cls_cv_best_params.items() if 'pca' in i}
        params = {'sc': None,
                  'pca': pca_best_param_dict,
                  'logistic': logistic_best_param_dict}
        cls_pipe = self._pipelinedef(params)
        cls_reg = cls_pipe.fit(x, y > 0)
        return cls_reg
