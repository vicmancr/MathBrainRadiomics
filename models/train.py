import os, csv
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


def train_test_preparation(X, y, cv, use_pca, num_feat, test_size=0.2):
    '''
    Apply Standard Scaler when desired.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=cv)

    scaler = StandardScaler()
    if use_pca == 'norm':
        # Standardizing features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif use_pca == 'pca':
        # Standardizing features + PCA
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pca = PCA(n_components=3)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_random_forest(X, y, reprs, use_pca='no', num_feat=0, n_estimators=100, 
                        max_depth=5, seed=0, cv=0, extract_feat=False):
    '''
    Random forest model. Bagging by default (with max_feature=n_features).
    '''
    X_train, X_test, y_train, y_test = train_test_preparation(X, y, cv, use_pca, num_feat)

    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Saving feature importance
    feat_imp = None
    if extract_feat and X.shape[1] > 1: # Only when more than one feature is considered
        importances = regressor.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = np.asarray([v['lead'] for k,v in reprs.items()] + ['age_at_MRI'])
        
        feat_imp = pd.DataFrame(np.zeros((indices.shape[0],2)), columns=['name', 'importance'])
        for f in range(indices.shape[0]):
            feat_imp.iloc[f] = [ 
                names[indices[f]],
                importances[indices[f]]
            ]

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, feat_imp


def train_svm(X, y, reprs, use_pca='no', num_feat=0, **args):
    '''
    Suppor vector machine model.
    '''
    cv = args['cv']
    X_train, X_test, y_train, y_test = train_test_preparation(X, y, cv, use_pca, num_feat)

    regressor = SVR(C=1.0, epsilon=0.5)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, None


def train_lasso(X, y, reprs, use_pca='no', num_feat=0, **args):
    '''
    LASSO model.
    '''
    cv = args['cv']
    X_train, X_test, y_train, y_test = train_test_preparation(X, y, cv, use_pca, num_feat)

    regressor = Lasso(alpha=0.5, max_iter=1e5)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, None


def train_bagging_regressor(X, y, reprs, use_pca='no', num_feat=0, n_estimators=100, max_depth=5, seed=0, cv=0):
    X_train, X_test, y_train, y_test = train_test_preparation(X, y, cv, use_pca, num_feat)

    regressor = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=max_depth, random_state=seed), 
        n_estimators=n_estimators, random_state=seed
    )
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    diff_abs = np.abs(y_test - y_pred)
    return diff_abs, y_test, None


def run_experiment_one_fold(model, path_to_results, data, radf, outcome, cv, use_age=False, use_pca='no', 
                            num_features=0, num_models=1, use_representatives=True, extract_features=False):
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
    regions_interest = {
        1: [11123, 11125, 11133, 11108, 11103], # Concepts
        2: [12131, 12113, 11153, 11115, 16], # AppliedProblems
        3: [12140, 11157, 12154, 12128, 11154], # MathFluency
        4: [26, 12114, 11108, 12171] # Calculus
    }
    labels = set([int(s[0][2:]) for s in radf.columns.str.split('_')[1:]])
    auxDF = pd.DataFrame(np.zeros((len(labels), 6)), columns=['mae', 'std', 'mse', 'std_mse', 'mape', 'label'])
    for idx, lb in tqdm(enumerate(sorted(labels)), total=len(labels), desc='Exp. for all labels', unit='lb'):
        if extract_features:
            print('#'*60)
            print('  Warning: Do not extract features for all regions and all models.')
            print('           Otherwise, you will end up with a huge collection of files.')
            print('#'*60)
            if lb not in regions_interest[outcome]: continue

        auxACC = pd.DataFrame(np.zeros((num_models, 5)), columns=['mae', 'std', 'mse', 'std_mse', 'mape'])
        for j in range(num_models):
            subdf = get_data(radf, lb)
            nsubdf = subdf.dropna(axis=1, how='any')
            reprs = get_representatives(nsubdf)
            X, y = get_X_y(data, nsubdf, reprs, outcome=outcome, use_reprs=use_representatives)

            # Save representatives to file
            if extract_features:
                os.makedirs(os.path.join(path_to_results, 'fi'), exist_ok=True)
                with open(os.path.join(path_to_results, 'fi', 'representatives_lb{0}_cv{1:02d}_{2}.csv'.format(lb, cv, outcome)), 'w') as f:
                    w = csv.DictWriter(f, reprs.keys())
                    w.writeheader()
                    w.writerow(reprs)

            # Drop last column (age) when modelling without age
            auxX = X[:] if use_age else X[:,:-1]

            # Train model (explicit setting of seeds for reproducibility)
            diff, y_test, feat_imp = train_model[model](auxX, y, reprs, use_pca, num_features, 
                                                        seed=j, cv=cv, extract_feat=extract_features)
            ym = y.mean()
            auxACC.iloc[j] = [
                np.average(diff)/ym, 
                np.std(diff)/ym,
                np.average(np.power(diff, 2))/(ym**2), 
                np.std(np.power(diff, 2))/(ym**2),
                np.average( np.divide(diff, np.abs(y_test)) )
            ]

            if feat_imp is not None:
                feat_imp.to_csv(os.path.join(
                    path_to_results, 'fi', 
                    'feature_importance_lb{0}_cv{1:02d}_rf{2:03d}_{3}.csv'.format(lb, cv, j, outcome))
                )

        auxDF.iloc[idx] = [*auxACC.mean(), lb]

    return auxDF


def age_model(model, dataf, radf, outcome, cv=0, num_models=1):
    # Set of models with different random seeds with only AGE as input
    train_model = {
        'rf': train_random_forest,
        'bag': train_bagging_regressor,
        'svm': train_svm,
        'lasso': train_lasso
    }
    results = pd.DataFrame(np.zeros((num_models, 5)), columns=['mae', 'std', 'mse', 'std_mse', 'mape'])
    lb = 12134
    subdf = get_data(radf, lb)
    nsubdf = subdf.dropna(axis=1, how='any')
    X, y = get_X_y(dataf, nsubdf, None, outcome=outcome, use_reprs=False)
    for j in tqdm(range(num_models), desc='Age model'):
            # Models with age
            diff, y_test, _ = train_model[model](X[:,-1].reshape(-1, 1), y, None, seed=j, cv=cv)
            ym = y.mean()
            results.iloc[j] = [
                np.average(diff)/ym, 
                np.std(diff)/ym,
                np.average(np.power(diff, 2))/(ym**2), 
                np.std(np.power(diff, 2))/(ym**2),
                np.average( np.divide(diff, np.abs(y_test)) )
            ]

    return results