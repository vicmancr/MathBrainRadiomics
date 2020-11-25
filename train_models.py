import re, os
import argparse
import numpy as np
import pandas as pd

from models.train import run_experiment_one_fold, age_model


def main(wd, path_to_data, cv, model, data_suffix, preprocessing, num_models, num_feat, all_radiomics, important_feat):
    rad = pd.read_csv(os.path.join(path_to_data, 'sample_radiomics_{}.csv'.format(data_suffix)))
    rad = rad[rad.userid != 0]

    data = pd.read_csv(os.path.join(path_to_data, 'sample_data.csv'))

    # Select users present in both datasets
    users = set(data['userid']).intersection(rad.userid)
    radf = rad.query('userid in @users')
    dataf = data.query('userid in @users')

    tag = '_' + preprocessing + str(num_feat) if preprocessing in ['pca'] else ''
    names = ['Concepts', 'AppliedProblems', 'MathFluency', 'Calculus']
    path_to_results = os.path.join(wd, 'results', 'cv_folds_{}_{}{}'.format(model, data_suffix, tag))
    os.makedirs(path_to_results, exist_ok=True)
    print('CV{0:02d}'.format(cv))
    for o,n in enumerate(names):
        o += 1
        print(o,n)

        # Filter dataset (in case of missing radiomics values)
        subdataf = dataf[~pd.isnull(dataf[n])]
        users = subdataf.userid
        subradf = radf.query('userid in @users')

        if os.path.exists(os.path.join(path_to_results, 'age_model_cv{0:02d}_{1}_{2}.csv'.format(cv, o, n))):
            results = pd.read_csv(os.path.join(path_to_results, 'age_model_cv{0:02d}_{1}_{2}.csv'.format(cv, o, n)))
        else:
            results = age_model(model, subdataf, subradf, o, cv, (num_models-1)//2+1)
            results.to_csv(os.path.join(path_to_results, 'age_model_cv{0:02d}_{1}_{2}.csv'.format(cv, o, n)))
        if os.path.exists(os.path.join(path_to_results, 'experiment_w_age_cv{0:02d}_{1}_{2}.csv'.format(cv, o, n))):
            df = pd.read_csv(os.path.join(path_to_results, 'experiment_w_age_cv{0:02d}_{1}_{2}.csv'.format(cv, o, n)))
        else:
            df = run_experiment_one_fold(model, path_to_results, subdataf, subradf, o, cv, 
                                         use_age=True, 
                                         use_pca=preprocessing, 
                                         num_features=num_feat,
                                         num_models=num_models,
                                         use_representatives=not all_radiomics,
                                         extract_features=important_feat)
            df.to_csv(os.path.join(path_to_results, 'experiment_w_age_cv{0:02d}_{1}_{2}.csv'.format(cv, o, n)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ensembles for each brain region.')
    parser.add_argument('cv', type=int, help='The cv partition to compute.')
    parser.add_argument('model', type=str, help='The model to use [rf, bag, svm].')
    parser.add_argument('data', type=str, help='The dataset to use [destrieux or random].')
    parser.add_argument('pre', type=str, help='Pre-processing method to use [norm, pca].')
    parser.add_argument('--num_models', type=int, default=100, 
                        help='Number of models to use and average over.')
    parser.add_argument('--num_feat', type=int, default=10, 
                        help='Number of features to use on pre-processing. Only for PCA.')
    parser.add_argument('--all_radiomics', type=bool, default=False, help='Use all radiomics?')
    parser.add_argument('--important_feat', type=bool, default=False, 
                        help='Extract important features for each model. Only available for Random Forest.')
    args = parser.parse_args()

    # Working directory
    wd = os.path.dirname(os.path.realpath(__file__))
    path_to_data = os.path.join(wd, 'data')

    main(wd, path_to_data, args.cv, args.model, 
        args.data, args.pre, args.num_models, 
        args.num_feat, args.all_radiomics, args.important_feat)