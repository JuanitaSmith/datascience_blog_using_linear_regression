import os
import time

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_selection import mutual_info_regression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split


def reduce_mem_usage(df):
    """ iterate through all the numerical columns of a dataframe and modify the data type
        to reduce memory usage.
    """

    print('\nTriggering memory optimization.......\n')

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type != 'boolean' and col_type != 'bool':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def create_folder(folder_name):
    """ Make directory if it doesn't already exist """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def change_width(ax, new_value):
    """ change width of bar plot"""
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def fit_imputer(df, tolerance=0.2, verbose=2, max_iter=20, nearest_features=20, imputation_order='ascending',
                initial_strategy='most_frequent'):
    """
    A function to train an IterativeImputer using machine learning

    Args:
        df: dataset to impute
        tolerance: Tolerance of stopping function
        verbose: Verbosy flag, controls the debug messages that are issued as functions are evaluated
        max_iter: Maximum number of imputation rounds
        nearest_features: Number of other features to use to estimate the missing values
        imputation_order: ascending or descending - the order in which the features will be imputed
        initial_strategy: e.g. 'most_frequent' or 'mean'

    Returns: dataset with no missing values

    """

    start = time.time()

    # restrict the values to be predicted to a min / max range
    minimum_before = list(df.iloc[:, :].min(axis=0))
    maximum_before = list(df.iloc[:, :].max(axis=0))
    print(minimum_before)
    print(maximum_before)

    imputer = IterativeImputer(random_state=0,
                               imputation_order=imputation_order,
                               n_nearest_features=nearest_features,
                               initial_strategy=initial_strategy,
                               max_iter=max_iter,
                               min_value=minimum_before,
                               max_value=maximum_before,
                               skip_complete=True,
                               tol=tolerance,
                               verbose=verbose)

    imputer_df = imputer.fit_transform(df)

    end = time.time()
    print('Execution time for IterativeImputer: {} sec'.format(end - start))

    return imputer_df


# Use entity recognition from natural language tools to automatically classify a person or business


def entity_recognision(name, nlp):
    """ detect if name is a person's name or an organization """

    # keywords to classify as a business to help the algorithm a little
    business_key_words = ['hotel', 'hotels', 'hostel', 'hostels', 'executive', 'service', 'travelnest', 'co', 'family',
                          'the', 'underthedoormat', 'destination8', 'guest', 'international', 'londonflats', 'property',
                          'homes', 'SynthAccommodation', 'accommodation', 'apartments']

    # keywords to classify as a person.
    # Sometimes algorithm cannot determine it's a person when & or and are inside e.g. 'Doreen & Nichola'
    individual_key_words = ['&', 'and']

    doc = nlp(name)
    ent = [ent.type for sent in doc.sentences for ent in sent.ents]

    # sometimes different words of a host name are classified more than once, e.g. ['PERSON', 'ORG']
    if ('ORG' in ent) or ('GPE' in ent) or ('FAC' in ent) or ('CARDINAL' in ent):
        ent = 'ORG'
    else:
        ent = 'PERSON'

    # let's help a little bit the model it's not always perfect
    for n in name.split(' '):
        if n.lower() in business_key_words:
            ent = 'ORG'
            break
        elif n.lower() in individual_key_words:
            ent = 'PERSON'
            break

    return ent


def log_transform(x):
    """ Convert variable to log10 scale. """

    return np.log10(x)


def inverse_log_transform(x):
    """ Convert variable back from log10 to original scale. """

    return sp.special.exp10(x)


def get_mi_score(X, y):
    """
    Calculate mutual information to show feature importance.

    Estimated mutual information between each feature and the target

    Args:
        X (array-like or sparse matrix) - Independent features
        y (array-like) - Dependent feature to be predicted
    Returns:
        mi (ndarray) - estimated mutual information
    """

    mi = mutual_info_regression(X, y, random_state=10)
    mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi


def calculate_residuals(model, X, y):
    """
    Calculate residuals (actual - predicted)

    Calculate predictions using fitted model object and independent variables.
    Use predictions to calculate residuals.

    Args:
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        model - scikit-learn fitted model
    Returns
        df_results (pandas dataframe) -  with columns 'Residuals',
                                        'Actual', 'Predicted'
    """

    predictions = model.predict(X)
    df_results = pd.DataFrame({'Actual': y, 'Predicted': predictions})
    df_results['Residuals'] = \
        abs(df_results['Actual']) - abs(df_results['Predicted'])

    return df_results


# build a final cleaning function
def cleaning(df, cluster_preprocessor, cluster_model, onehot_preprocessor, random_state=0):
    """
    Cleaning function to prepare listing dataset for modelling.

    Args:
        df (pandas dataframe) - listing dataset cleaned by notebooks 1 and 2
        cluster_preprocesser - pre-fitted model for clustering preparation
        cluster_model - pre-fitted model to create property_cluster variable
        onehot_preprocessor - pre-fitted model for onehot encoding
    """
    # cluster preprocess for only relevant columns
    df_filtered = df[['room_type',
                      'property_type',
                      'latitude',
                      'longitude',
                      'price_mean']].copy(deep=True)

    df_transformed = cluster_preprocessor.transform(df_filtered)

    # cluster data with already fitted model
    df_transformed['cluster'] = cluster_model.predict(df_transformed)

    # rank and renaming clusters to be in sequential order
    tmp = df_transformed.groupby('cluster').agg(
        min=('price_mean', 'min'),
        max=('price_mean', 'max'),
        nr_listings=('cluster', 'count')).sort_values(by='min')

    tmp['rank'] = tmp['min'].rank(method='max').astype('int')

    # resequence the cluster names by price
    mymap = pd.Series(tmp['rank'].values, index=tmp.index).to_dict()

    # rename clusters to sequence them into ranked order
    df_transformed['property_cluster'] = df_transformed['cluster'].map(mymap)

    # add cluster to main dataset
    df['property_cluster'] = df_transformed['property_cluster']

    # drop lat/long after clustering
    df = df.drop(['latitude', 'longitude'], axis=1)

    # split the dataset into dependent and independent datasets
    y = df['price_mean'].copy()
    X = df.drop('price_mean', axis=1)

    # final onehot encoding for all features dropping sparse columns
    X_transformed = onehot_preprocessor.transform(X)

    # make sure property types are well presented
    # in both training and testing sets using stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=X_transformed[['property_cluster',
                                'is_business_True']],
    )

    return X_train, X_test, y_train, y_test