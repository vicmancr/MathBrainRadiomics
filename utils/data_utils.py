import numpy as np


def get_data(df, label):
    '''
    Return a subset of columns of provided dataframe by label.
    Params:
        df: Pandas DataFrame with columns following the pattern 
            lbXX_radiomics_feature.
        label: Integer value for desired label.
    Returns:
        Pandas DataFrame
    '''
    subdf = df.loc[:, df.columns.str.contains('lb{}_'.format(label))]

    return subdf


def get_representatives(df, thrs=0.8):
    '''
    Select representative features based on correlation groups.
    Choose one feature and group with it all features that have
    a correlation squared greater or equal than a given threshold. 
    Params:
        df: Pandas DataFrame.
        thrs: (optional) Correlation threshold to consider when 
            selecting groups.
    Returns:
        Dict with representative of group (lead), index (leadIdx)
        and members of the group (members).
    '''
    auxcorr = df.corr()
    ordering = []
    reprs = {}
    rsquared = auxcorr**2
    gps = 0
    for idx, cl in enumerate(auxcorr.columns):
        if idx in ordering:
            continue
        column = rsquared.iloc[idx].values
        order = np.argsort(column)[::-1]
        ordered = column[order]
        indices = order[ordered > thrs]
        unq = set(indices) - set(ordering)
        reprs[gps] = {
            'lead': cl, 'leadIdx': idx, 
            'members': list(unq)
        }
        ordering.extend(unq)
        gps += 1

    return reprs


def get_X_y(data, subdf, reprs, outcome=4, use_reprs=True):
    '''
    Return X and y data arrays to use as input and output,
    respectively, for a model.
    Params:
        data: Pandas DataFrame with subjects information and 
            results in different tasks.
        subdf: Pandas DataFrame contained desired label.
            This is a subset of the radiomics DataFrame.
        reprs: Dictionary of representative features obtained
            according to feature correlations.
        outcome: Integer value pointing to the index of the outcome
            to choose.
        use_reprs: Whether reprs are considered or not.
    Returns:
        Two numpy arrays serving as X and y for the model.
    '''
    if use_reprs:
        leads = [v['lead'] for k,v in reprs.items()]
        aux = subdf.loc[:, np.asarray(leads)].values
    else:
        aux = subdf.values
    X = np.zeros((aux.shape[0], aux.shape[1]+5))
    X[:,:-5] = aux
    X[:, -5] = data['age_at_MRI'].values
    names = ['Concepts', 'AppliedProblems', 'MathFluency', 'Calculus']
    for i,n in enumerate(names):
        i += 1
        X[:,-i] = data[n].values

    return X[:,:-4], X[:,-outcome]