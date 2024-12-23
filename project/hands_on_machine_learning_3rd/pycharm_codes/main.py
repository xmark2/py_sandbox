import sys
import os
import numpy as np
import pandas as pd
import tarfile
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
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


def explore_data(data):
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
    print('-' * 10)
    print('SHAPE')
    print(data.shape)

    print('-'*10)
    print('HEAD')
    print(data.head())

    print('-' * 10)
    print('INFO')
    print(data.info())

    print('-' * 10)
    print('Value_Counts')
    for col in data.columns:
        print(data[col].value_counts())

    print('-' * 10)
    print('Describe')
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
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    return train_set, test_set


def prepare_data(train_set):
    """
    Prepare the training set for machine learning by handling missing values, scaling numerical features, and encoding categorical features.

    Parameters:
    train_set (DataFrame): The training set to prepare.

    Returns:
    tuple: A tuple containing the prepared features and the labels as NumPy arrays.
    """
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = list(housing.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, housing_labels


def train_model(housing_prepared, housing_labels):
    """
    Train a linear regression model on the prepared training data.

    Parameters:
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    LinearRegression: The trained linear regression model.
    """
    model = LinearRegression()
    model.fit(housing_prepared, housing_labels)
    return model


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
    mse = mean_squared_error(housing_labels, predictions)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")


def execute():
    housing_data = download_housing_data()

    housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    explore_data(housing_data)

    # train_set, test_set = split_data(housing_data)
    # housing_prepared, housing_labels = prepare_data(train_set)
    #
    # model = train_model(housing_prepared, housing_labels)
    #
    # evaluate_model(model, housing_prepared, housing_labels)

# def matplot():
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # Generate some data
#     x = np.linspace(0, 10, 100)
#     y = np.sin(x)
#
#     # Create a plot
#     plt.plot(x, y)
#     plt.title('Sine Wave')
#     plt.xlabel('X axis')
#     plt.ylabel('Y axis')
#
#     # Show the plot
#     plt.show()


# Main execution
if __name__ == "__main__":
    # data_path = os.path.join("datasets", "housing", "housing.csv")
    # housing_data = load_data(data_path)
    execute()
    # matplot()
