import numpy as np
import pandas as pd
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


## Option1
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


## Option2
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio=0.2, id_column=None):
    if not id_column:
        data = data.reset_index()
        id_column = 'index'
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

## Option 3
def split_data_sklearn_train_test_split(data, test_size=0.2, random_state=42):
    '''

    :param data:
    :param test_size:
    :param random_state:
    :return:
    example: train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    '''
    return train_test_split(data, test_size=test_size, random_state=random_state)

## Option 4
def split_data_strat(data, n_splits=10, test_size=0.2, random_state=42, stratify_col='income_cat'):
    '''

    :param data:
    :param n_splits:
    :param test_size:
    :param random_state:
    :param stratify_col:
    :return:
    '''

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    strat_splits = []
    for train_index, test_index in splitter.split(data, data[stratify_col]):
        strat_train_set_n = data.iloc[train_index]
        strat_test_set_n = data.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])

    return strat_splits

## Option 4 short version
def stratified_train_test_split(data, test_size=0.2, random_state=42, stratify_col='income_cat'):
    """
    Perform a stratified train-test split on the given dataset based on the specified stratify column.

    Parameters:
    data (DataFrame): The dataset to split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Seed used by the random number generator for reproducibility.
    stratify_col (str): The column to use for stratification.

    Returns:
    tuple: A tuple containing the training and test sets as pandas DataFrames.
    """
    strat_train_set, strat_test_set = train_test_split(
        data, test_size=test_size, stratify=data[stratify_col], random_state=random_state
    )
    return strat_train_set, strat_test_set
