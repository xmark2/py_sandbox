import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.compose import make_column_selector, make_column_transformer

from sklearn.preprocessing import FunctionTransformer


def preprocessed_to_df(housing_prepared, preprocessing, housing):
    # extra code – shows that we can get a DataFrame out if we want
    housing_prepared_fr = pd.DataFrame(
        housing_prepared,
        columns=preprocessing.get_feature_names_out(),
        index=housing.index)
    print(housing_prepared_fr.head(2))
    return housing_prepared_fr

def prepare_data_option1(train_set):
    """
    Prepare the training set for machine learning by handling missing values, scaling numerical features, and encoding categorical features.

    Parameters:
    train_set (DataFrame): The training set to prepare.

    Returns:
    tuple: A tuple containing the prepared features and the labels as NumPy arrays.
    """
    # housing_num = housing.select_dtypes(include=[np.number])
    # text_data = housing.select_dtypes(include=['object'])

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    # num_pipeline Option1
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

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

    housing_prepared = preprocessing.fit_transform(housing)

    preprocessed_to_df(housing_prepared, preprocessing, housing)

    return housing_prepared, housing_labels

def prepare_data_option2(train_set):
    """
    Prepare the training set for machine learning by handling missing values, scaling numerical features, and encoding categorical features.

    Parameters:
    train_set (DataFrame): The training set to prepare.

    Returns:
    tuple: A tuple containing the prepared features and the labels as NumPy arrays.
    """

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))



    # preprocessing Option2
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )

    housing_prepared = preprocessing.fit_transform(housing)

    preprocessed_to_df(housing_prepared, preprocessing, housing)

    return housing_prepared, housing_labels



def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


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




def prepare_data_option3(train_set):
    """
    Prepare the training set for machine learning by handling missing values, scaling numerical features, and encoding categorical features.

    Parameters:
    train_set (DataFrame): The training set to prepare.

    Returns:
    tuple: A tuple containing the prepared features and the labels as NumPy arrays.
    """

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

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

