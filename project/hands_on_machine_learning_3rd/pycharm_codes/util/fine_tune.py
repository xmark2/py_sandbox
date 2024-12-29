import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint


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


def grid_search_fine_tune(housing_prepared, housing_labels):
    """
    Fine-tune a model using GridSearchCV to find the best hyperparameters.

    Parameters:
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    RandomForestRegressor: The best estimator found by GridSearchCV.
    """
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    print(f"Best Parameters (Grid Search): {grid_search.best_params_}")
    print(f"Best Estimator (Grid Search): {grid_search.best_estimator_}")

    return grid_search.best_estimator_


def randomized_search_fine_tune(housing_prepared, housing_labels):
    """
    Fine-tune a model using RandomizedSearchCV to find the best hyperparameters.

    Parameters:
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    RandomForestRegressor: The best estimator found by RandomizedSearchCV.
    """
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor()
    rnd_search = RandomizedSearchCV(forest_reg, param_distribs, n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                    random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)

    print(f"Best Parameters (Randomized Search): {rnd_search.best_params_}")
    print(f"Best Estimator (Randomized Search): {rnd_search.best_estimator_}")

    return rnd_search.best_estimator_


def ensemble_methods(housing_prepared, housing_labels):
    """
    Combine multiple models using ensemble methods to improve performance.

    Parameters:
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    VotingRegressor: The ensemble model combining multiple estimators.
    """
    forest_reg = RandomForestRegressor(n_estimators=30, max_features=8)
    boosting_reg = GradientBoostingRegressor()

    ensemble_model = VotingRegressor(estimators=[('rf', forest_reg), ('gb', boosting_reg)])
    ensemble_model.fit(housing_prepared, housing_labels)

    print(f"Ensemble Model: {ensemble_model}")

    return ensemble_model


def analyze_best_models(grid_search, rnd_search, housing_prepared, housing_labels):
    """
    Analyze the best models and their errors to understand performance.

    Parameters:
    grid_search (GridSearchCV): The GridSearchCV object after fitting.
    rnd_search (RandomizedSearchCV): The RandomizedSearchCV object after fitting.
    housing_prepared (ndarray): The prepared features.
    housing_labels (Series): The labels.

    Returns:
    None
    """
    best_grid_model = grid_search.best_estimator_
    best_rnd_model = rnd_search.best_estimator_

    print("Feature Importances from Grid Search:")
    feature_importances = best_grid_model.feature_importances_
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted_features = sorted(zip(feature_importances, attributes), reverse=True)
    for importance, feature in sorted_features:
        print(f"{feature}: {importance}")

    print("Analyzing Residual Errors for Grid Search:")
    evaluate_model(best_grid_model, housing_prepared, housing_labels)

    print("Analyzing Residual Errors for Randomized Search:")
    evaluate_model(best_rnd_model, housing_prepared, housing_labels)

#
# # Main execution
# if __name__ == "__main__":
#     data_path = os.path.join("datasets", "housing", "housing.csv")
#     housing_data = load_data(data_path)
#
#     explore_data(housing_data)
#
#     train_set, test_set = split_data(housing_data)
#
#     housing_prepared, housing_labels = prepare_data(train_set)
#
#     model = train_model(housing_prepared, housing_labels)
#
#     evaluate_model(model, housing_prepared, housing_labels)
#
#     best_grid_model = grid_search_fine_tune(housing_prepared, housing_labels)
#     best_rnd_model = randomized_search_fine_tune(housing_prepared, housing_labels)
#     ensemble_model = ensemble_methods(housing_prepared, housing_labels)
#
#     analyze_best_models(best_grid_model, best_rnd_model, housing_prepared, housing_labels)
