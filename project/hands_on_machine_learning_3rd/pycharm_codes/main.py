import sys
import os
import numpy as np
import pandas as pd
import tarfile
from pathlib import Path
import urllib.request
from util import split
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


# In[96]:


from sklearn.cluster import KMeans


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]



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
    # housing_num = housing.select_dtypes(include=[np.number])

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    # num_pipeline Option1
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    from sklearn.pipeline import make_pipeline

    # num_pipeline Option2
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    # preprocessing Option1
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                   "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    from sklearn.compose import make_column_selector, make_column_transformer

    # preprocessing Option2
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )

    housing_prepared = preprocessing.fit_transform(housing)

    # extra code – shows that we can get a DataFrame out if we want
    housing_prepared_fr = pd.DataFrame(
        housing_prepared,
        columns=preprocessing.get_feature_names_out(),
        index=housing.index)
    print(housing_prepared_fr.head(2))

    from sklearn.preprocessing import FunctionTransformer

    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return ["ratio"]  # feature names out

    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler())

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler())

    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        # ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
        remainder=default_num_pipeline)  # one column remaining: housing_median_age

    housing_prepared = preprocessing.fit_transform(housing)

    # extra code – shows that we can get a DataFrame out if we want
    housing_prepared_fr = pd.DataFrame(
        housing_prepared,
        columns=preprocessing.get_feature_names_out(),
        index=housing.index)
    print(housing_prepared_fr.head(2))

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

    # print(housing_data.select_dtypes(include=[np.number]).columns)
    # text_data = housing_data.select_dtypes(include=['object'])
    # print(np.number)

    explore_data(housing_data, inf='info')

    train_set, test_set = split_data(housing_data)
    housing_prepared, housing_labels = prepare_data(train_set)
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
