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
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32


def split_data_with_id_hash(data, test_ratio=0.2, id_column=None):
    if not id_column:
        data = data.reset_index()
        id_column = 'index'
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


## Option 3
def split_train_test_split_random(data, test_size=0.2, random_state=42):
    '''

    :param data:
    :param test_size:
    :param random_state:
    :return:
    example: train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    '''
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_set, test_set

def data_add_strat_col(data, strat_dict):
    stratify_col = strat_dict['stratify_col']
    stratify_col_source = strat_dict['stratify_col']
    bins = strat_dict['bins']
    labels = strat_dict['labels']

    data[stratify_col] = pd.cut(data[stratify_col_source], bins=bins, labels=labels)
    return data


## Option 4
def split_data_strat(data, strat_dict, n_splits=10, test_size=0.2, random_state=42):
    """
    Perform a stratified train-test split on the given dataset based on the specified stratify column.

    Parameters:
    data (DataFrame): The dataset to split.
    stratify_col (str): The column to use for stratification. Default is 'median_income'.
    n_splits (int): The number of re-shuffling and splitting iterations. Default is 10.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    random_state (int): Seed used by the random number generator for reproducibility. Default is 42.

    Returns:
    list: A list containing tuples of (train_set, test_set) for each split.
    """
    stratify_col = strat_dict['stratify_col']
    data = data_add_strat_col(data, strat_dict)

    # Binning the continuous stratify column if it's 'median_income'
    if not stratify_col and stratify_col not in data.columns:
        return

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    strat_splits = []
    for train_index, test_index in splitter.split(data, data[stratify_col]):
        strat_train_set_n = data.iloc[train_index]
        strat_test_set_n = data.iloc[test_index]
        strat_splits.append((strat_train_set_n, strat_test_set_n))

    # Drop the temporary 'income_cat' column if it was created
    if stratify_col in data.columns:
        for strat_train_set_n, strat_test_set_n in strat_splits:
            strat_train_set_n.drop(stratify_col, axis=1, inplace=True)
            strat_test_set_n.drop(stratify_col, axis=1, inplace=True)

    return strat_splits


## Option 4 short version
def split_data_strat_short(data, strat_dict, test_size=0.2, random_state=42):
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
    stratify_col = strat_dict['stratify_col']
    data = data_add_strat_col(data, strat_dict)

    if not stratify_col and stratify_col not in data.columns:
        return

    strat_train_set, strat_test_set = train_test_split(
        data, test_size=test_size, stratify=data[stratify_col], random_state=random_state
    )

    # Drop the temporary 'income_cat' column if it was created
    for set_ in (strat_train_set, strat_test_set):
        set_.drop(stratify_col, axis=1, inplace=True)

    return strat_train_set, strat_test_set


## overall test results
def income_cat_proportions(data, strat_dict):
    stratify_col = strat_dict['stratify_col']
    return data[stratify_col].value_counts() / len(data)


def overall_test(data, strat_dict):
    strat_train_set, strat_test_set = split_data_strat_short(data, strat_dict)
    train_set, test_set = split_train_test_split_random(data)

    compare_props = pd.DataFrame({
        "Overall %": income_cat_proportions(data, strat_dict),
        "Stratified %": income_cat_proportions(strat_test_set, strat_dict),
        "Random %": income_cat_proportions(test_set, strat_dict),
    }).sort_index()

    compare_props.index.name = "Income Category"
    compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                       compare_props["Overall %"] - 1)
    compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                      compare_props["Overall %"] - 1)

    (compare_props * 100).round(2)
    return compare_props
