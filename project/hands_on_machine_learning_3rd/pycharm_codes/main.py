import sys
import os
import numpy as np
import pandas as pd
import tarfile
from pathlib import Path
import urllib.request
from util import split, prepare, fine_tune
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error


## Download the Data
def download_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def explore_data(data, inf):
    """
    Display the first few rows of the dataset and plot histograms of all features.

    Parameters:
    data (DataFrame): The dataset to explore.

    Returns:
    None
    """
    if not hasattr(data, 'head'):
        return

    pd.options.plotting.backend = "matplotlib"
    # print('-' * 10)
    if str(inf).lower().startswith('shape'):
        # print('SHAPE')
        print(data.shape)

    # print('-'*10)
    if str(inf).lower().startswith('head'):
        # print('HEAD')
        print(data.head())

    # print('-' * 10)
    # print('INFO')
    if str(inf).lower().startswith('info'):
        print(data.info())

    # print('-' * 10)
    # print('Value_Counts')
    if str(inf).lower().startswith('val'):
        for col in data.columns:
            print(data[col].value_counts())

    # print('-' * 10)
    # print('Describe')
    if str(inf).lower().startswith('desc'):
        print(data.describe())

    # data.hist(bins=50, figsize=(20, 15))
    # plt.show()


def split_data(data):
    """
    Split the dataset into a training set and a test set.

    Parameters:
    data (DataFrame): The dataset to split.

    Returns:
    tuple: A tuple containing the training set and the test set as pandas DataFrames.
    """
    # train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    train_set, test_set = split.split_train_test_split_random(data)
    return train_set, test_set


def prepare_data(train_set):
    """
    Prepare the training set for machine learning by handling missing values, scaling numerical features, and encoding categorical features.

    Parameters:
    train_set (DataFrame): The training set to prepare.

    Returns:
    tuple: A tuple containing the prepared features and the labels as NumPy arrays.
    """

    # return prepare.prepare_data_option1(train_set=train_set)
    # return prepare.prepare_data_option2(train_set=train_set)
    return prepare.prepare_data_option3(train_set=train_set)

def train_model(housing_prepared, housing_labels):
    """
    Train a linear regression model on the prepared training data.

    Parameters:
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    LinearRegression: The trained linear regression model.
    """
    # model = LinearRegression()
    model = DecisionTreeRegressor(random_state=42)
    # model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    return model

    # from sklearn.linear_model import LinearRegression
    #
    # lin_reg = make_pipeline(preprocessing, LinearRegression())
    # lin_reg.fit(housing, housing_labels)


def evaluate_model(model, housing_prepared, housing_labels):
    """
    Evaluate the trained model on the prepared training data and print the Root Mean Squared Error (RMSE).

    Parameters:
    model (LinearRegression): The trained linear regression model.
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    None
    """
    predictions = model.predict(housing_prepared)
    print('predictions', predictions[:5].round(-2))
    print('housing_labels', housing_labels.iloc[:5].values)

    # extra code – computes the error ratios discussed in the book
    error_ratios = predictions[:10].round(-2) / housing_labels.iloc[:10].values - 1
    print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

    mse = mean_squared_error(housing_labels, predictions)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    # extra code – computes the error stats for the linear model
    # lin_rmses = -cross_val_score(lin_reg, housing, housing_labels,
    #                              scoring="neg_root_mean_squared_error", cv=10)
    # pd.Series(lin_rmses).describe()

    from sklearn.model_selection import cross_val_score

    tree_rmses = -cross_val_score(model, housing_prepared, housing_labels,
                                  scoring="neg_root_mean_squared_error", cv=10)
    print(pd.Series(tree_rmses).describe())


def fine_tune_options(model, housing_prepared, housing_labels):
    fine_tune.grid_search_fine_tune(housing_prepared, housing_labels)


def execute():
    housing_data = download_housing_data()

    explore_data(housing_data, inf='info')

    train_set, test_set = split_data(housing_data)
    housing_prepared, housing_labels = prepare_data(train_set)

    model = train_model(housing_prepared, housing_labels)
    evaluate_model(model, housing_prepared, housing_labels)
    fine_tune_options(model, housing_prepared, housing_labels)




# Main execution
if __name__ == "__main__":
    # data_path = os.path.join("datasets", "housing", "housing.csv")
    # housing_data = load_data(data_path)
    execute()
