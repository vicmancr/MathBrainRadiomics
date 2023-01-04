'Models'

import os
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.data_utils import *


def train_test_preparation(X, y, cv, use_reprs, use_pca, num_feat, test_size=0.2, **args):
    '''
    Apply Standard Scaler when desired.
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=cv)

    train_reprs = None
    if use_reprs:
        # Feature selection (only with train data wo. age)
        train_reprs = get_representatives(x_train.iloc[:, :-1])
        leads = [v['lead'] for k, v in train_reprs.items()]
        # Add age always to ensure its presence in final dataset
        leads.append('age_at_MRI')
        x_train = x_train.loc[:, np.asarray(leads)]
        x_test = x_test.loc[:, np.asarray(leads)]

        # Save representatives to file
        # if args['path_to_results'] != '':
        #     os.makedirs(os.path.join(args['path_to_results'], 'fi'), exist_ok=True)
        #     _filename = os.path.join(
        #         args['path_to_results'], 'fi',
        #         'representatives_lb{0}_cv{1:02d}_{2}.csv')
        #     with open(_filename.format(args['lb'], cv, args['outcome']), 'w') as _file:
        #         w = csv.DictWriter(_file, train_reprs.keys())
        #         w.writeheader()
        #         w.writerow(train_reprs)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    scaler = StandardScaler()
    if use_pca == 'norm':
        # Standardizing features
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif use_pca == 'pca':
        # Standardizing features + PCA
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        pca = PCA(n_components=num_feat)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

    return x_train, x_test, y_train, y_test, train_reprs


def train_random_forest(
        X, y, use_reprs, use_pca='no', num_feat=0, n_estimators=100,
        max_depth=5, seed=0, cv=0, path_to_results='', lb=0, **args):
    '''
    Random forest model. Bagging by default (with max_feature=n_features).
    '''
    x_train, x_test, y_train, y_test, reprs = train_test_preparation(
        X, y, cv, use_reprs, use_pca, num_feat,
        path_to_results=path_to_results, lb=lb,
        outcome=args['outcome'])

    regressor = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    # Saving feature importance
    feat_imp = None
    # if x_train.shape[1] > 1: # Only when more than one feature is considered
    #     importances = regressor.feature_importances_
    #     std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
    #     indices = np.argsort(importances)[::-1]
    #     names = np.asarray([v['lead'] for k, v in reprs.items()] + ['age_at_MRI'])

    #     feat_imp = pd.DataFrame(np.zeros((indices.shape[0], 2)), columns=['name', 'importance'])
    #     for f in range(indices.shape[0]):
    #         feat_imp.iloc[f] = [
    #             names[indices[f]],
    #             importances[indices[f]]
    #         ]

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, feat_imp


def train_svm(X, y, use_reprs, use_pca='no', num_feat=0, **args):
    '''
    Suppor vector machine model.
    '''
    cv = args['cv']
    x_train, x_test, y_train, y_test, _ = train_test_preparation(
        X, y, cv, use_reprs, use_pca, num_feat)

    regressor = SVR(C=1.0, epsilon=0.5)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, None


def train_lasso(X, y, use_reprs, use_pca='no', num_feat=0, **args):
    '''
    LASSO model.
    '''
    cv = args['cv']
    x_train, x_test, y_train, y_test, _ = train_test_preparation(
        X, y, cv, use_reprs, use_pca, num_feat)

    regressor = Lasso(alpha=0.5, max_iter=1e5)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, None


def train_bagging_regressor(
        X, y, use_reprs, use_pca='no', num_feat=0,
        n_estimators=100, max_depth=5, seed=0, cv=0):
    x_train, x_test, y_train, y_test, _ = train_test_preparation(
        X, y, cv, use_reprs, use_pca, num_feat)

    regressor = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=max_depth, random_state=seed),
        n_estimators=n_estimators, random_state=seed
    )
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, None


def run_experiment_one_fold(
        model, path_to_results, data, radf, outcome, cv,
        use_age=False, use_pca='no', num_features=0, num_models=1, use_representatives=True):
    '''
    Metrics are normalised mae (Mean Absolute Error) by outcome mean,
    normalised mse (Mean Squared Error) by outcome mean squared,
    mape (Mean Absolute Percentage Error)
    '''
    train_model = {
        'rf': train_random_forest,
        'bag': train_bagging_regressor,
        'svm': train_svm,
        'lasso': train_lasso
    }

    # Modify with your own regions of interest for
    # extracting important features for each of them.
    # regions_interest = {
    #     5: [11123, 11125, 11133], # Concepts
    #     6: [12131, 12113], # AppliedProblems
    #     7: [12140, 12154, 11157, 12128, 11154], # MathFluency
    #     8: [26, 12171, 11108, 12111, 12128] # Calculation
    # }
    labels = set([int(s[0][2:]) for s in radf.columns.str.split('_')[1:] if 'lb' in s[0]])
    aux_df = pd.DataFrame(
        np.zeros((len(labels), 6)),
        columns=['mae', 'std', 'mse', 'std_mse', 'mape', 'label'])
    for idx, lb in tqdm(enumerate(sorted(labels)), total=len(labels), desc='Exp. for all labels', unit='lb'):
        # if lb not in regions_interest[outcome]:
        #     continue
        aux_acc = pd.DataFrame(
            np.zeros((num_models, 5)),
            columns=['mae', 'std', 'mse', 'std_mse', 'mape'])
        for j in range(num_models):
            subdf = get_data(radf, lb)
            nsubdf = subdf.dropna(axis=1, how='any')
            X, y = get_X_y_series(data, nsubdf, outcome=outcome, use_reprs=use_representatives)

            # Drop last column (age) when modelling without age
            aux_x = X if use_age else X.drop(['age_at_MRI'], axis=1)

            # Train model (explicit setting of seeds for reproducibility)
            diff, y_test, feat_imp = train_model[model](
                aux_x, y, use_representatives, use_pca, num_features, seed=j, cv=cv,
                path_to_results=path_to_results, lb=lb, outcome=outcome)
            ym = y.mean()
            aux_acc.iloc[j] = [
                np.average(diff)/ym,
                np.std(diff)/ym,
                np.average(np.power(diff, 2))/(ym**2),
                np.std(np.power(diff, 2))/(ym**2),
                np.average(np.divide(diff, np.abs(y_test)))
            ]

            if feat_imp is not None:
                feat_imp.to_csv(os.path.join(
                    path_to_results, 'fi',
                    'feature_importance_lb{0}_cv{1:02d}_rf{2:03d}_{3}.csv'.format(
                        lb, cv, j, outcome)))

        aux_df.iloc[idx] = [*aux_acc.mean(), lb]

    return aux_df


def age_model(model, dataf, radf, outcome, cv=0, num_models=1):
    # Set of models with different random seeds with only AGE as input
    train_model = {
        'rf': train_random_forest,
        'bag': train_bagging_regressor,
        'svm': train_svm,
        'lasso': train_lasso
    }
    results = pd.DataFrame(
        np.zeros((num_models, 5)),
        columns=['mae', 'std', 'mse', 'std_mse', 'mape'])
    lb = 12134
    subdf = get_data(radf, lb)
    nsubdf = subdf.dropna(axis=1, how='any')
    X, y = get_X_y_series(dataf, nsubdf, outcome=outcome, use_reprs=False)
    for j in tqdm(range(num_models), desc='Age model'):
        # Models with age
        diff, y_test, _ = train_model[model](
            X['age_at_MRI'], y, None, seed=j, cv=cv, outcome=outcome)
        ym = y.mean()
        results.iloc[j] = [
            np.average(diff)/ym,
            np.std(diff)/ym,
            np.average(np.power(diff, 2))/(ym**2),
            np.std(np.power(diff, 2))/(ym**2),
            np.average(np.divide(diff, np.abs(y_test)))
        ]

    return results
