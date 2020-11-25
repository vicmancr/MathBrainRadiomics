import os
import argparse

import numpy as np
import pandas as pd

from scipy.stats import norm

def load_task(task_num, cv_num):
    '''
    Load and compute average results per task.
    Parameters:
        task_num: the task index in the range 1-8.
            Choose from 'Wrazonmate','Wdescal','Wbrevesmate','Wampliasmate',
                        'Concepts','Problems','Fluency','Calculus'
        cv_num: max number of partitions to consider.
    Returns:
        Mean results, mean noise results, median results, median noise results
        and age results, in order.
    '''
    cvs = np.arange(cv_num)
    texts = np.asarray(['Concepts','AppliedProblems','MathFluency','Calculus'])
    t = texts[task_num-1]
    data = {}
    noise_data = {}
    age_model = {}
    _path = './results'
    _folder = 'cv_folds_rf_destrieux'
    _name = 'experiment_w_age_cv{0:02d}_{1}_{2}.csv'
    _age = 'age_model_cv{0:02d}_{1}_{2}.csv'
    for cv in cvs:
        _f = _folder
        data['{0:02d}'.format(cv)] = pd.read_csv(os.path.join(_path, _f, _name.format(cv,task_num,t)))
        _f = '_'.join(_folder.split('_')[:-2]) + '_rf_random'
        noise_data['{0:02d}'.format(cv)] = pd.read_csv(os.path.join(_path, _f, _name.format(cv,task_num,t)))
        _f = _folder
        age_model['{0:02d}'.format(cv)] = pd.read_csv(os.path.join(_path, _f, _age.format(cv,task_num,t)))
    

    median_data = data['00'].copy()
    median_ndata = noise_data['00'].copy()

    _maes = np.asarray([data[k]['mae'].values for k in data.keys()])
    _nmaes = np.asarray([noise_data[k]['mae'].values for k in noise_data.keys()])

    median_data['mae'] = np.median(_maes, axis=0)
    median_data['std'] = _maes.std(axis=0)
    median_ndata['mae'] = np.median(_nmaes, axis=0)
    median_ndata['std'] = _nmaes.std(axis=0)

    rows = age_model['00'].shape[0]
    mean_age = pd.DataFrame(np.zeros((cvs.size*rows, 1)),columns=['mae'])
    _maes = np.asarray([age_model[k]['mae'].values for k in age_model.keys()])
    mean_age['mae'] = _maes.flatten()
    
    arg1 = (median_data, median_ndata)
    arg2 = mean_age

    return arg1, arg2


def run(cvs):
    '''
    Collect all results from the different partitions, extract medians and 
    compute p-values. Save results to csv files for each task.
    '''
    task = load_task(0, cvs)
    _cols = ['Concepts','AppliedProblems','MathFluency','Calculus']
    tasks = pd.DataFrame(index=task[0][0]['label'].astype(int), columns=_cols)

    for i in range(tasks.shape[1]):
        t = load_task(i+1, cvs)
        t_median = t[0]
        median = t_median[0]
        median.reset_index(drop=True, inplace=True)
        
        n_median = t_median[1]
        n_median.reset_index(drop=True, inplace=True)
        
        navg = n_median['mae'].mean()
        nstd = n_median['mae'].std() + 1e-8
        def prob(lb):
            return norm.cdf(median[median.label == lb]['mae'], loc=navg, scale=nstd)[0]
        
        avg = median['mae'].mean()
        std = median['mae'].std() + 1e-8
        def prob_rad(lb):
            return norm.cdf(median[median.label == lb]['mae'], loc=avg, scale=std)[0]

        # Significative regions according to noise distribution
        meddata_sorted = median.copy()
        meddata_sorted['prob_noise'] = list(map(prob, median.label))
        meddata_sorted['prob_rad'] = list(map(prob_rad, median.label))
        
        os.makedirs('./results/postprocess', exist_ok=True)
        meddata_sorted.sort_values(by='prob_noise').to_csv(
            './results/postprocess/significative_labels_median_{}_{}cvs.csv'.format(_cols[i], cvs),
            index=False
        )


def get_args():
    parser = argparse.ArgumentParser(description='Postprocess results from different partitions' \
                                    + 'and compute the level of significance for each area',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', type=int, default=20,
                        help='Number of partitions to consider for the median.', dest='cvs')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # python run.py -c 20
    run(args.cvs)
