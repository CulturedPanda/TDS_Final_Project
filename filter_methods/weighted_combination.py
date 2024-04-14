import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression


class WeightedCombination:
    """
    This class implements a filter method, grading the features
    based on a weighted combination of correlation and information gain.
    """

    def __init__(self, correlation_weight=0.5, target_column=None):
        self.mutual_information = None
        self.correlation = None
        self.correlation_weight = correlation_weight
        self.target_column = None

    def fit(self, X, target_column=None, continuous_cols=None, categorical_cols=None):
        if target_column is None and self.target_column is None:
            raise ValueError("Target column must be set")
        if target_column is not None:
            self.target_column = target_column
        # Compute the correlation between the features and the target
        # We use the absolute value of the correlation, as we are interested in the strength of the relationship,
        # regardless of the direction
        self.correlation = abs(pd.Series(X.corr().loc[:, self.target_column]).drop(index=self.target_column))

        # Isolate the target column
        y = X[self.target_column]
        data = X.drop(columns=[self.target_column])

        # If the user has only provided one of the lists, we will infer the other
        if continuous_cols is not None and categorical_cols is None:
            categorical_cols = [col for col in data.columns if col not in continuous_cols]
        elif continuous_cols is None and categorical_cols is not None:
            continuous_cols = [col for col in data.columns if col not in categorical_cols]

        # If both list were provided or at-least 1 was inferred, we will calculate the information gain
        # appropriately for the continuous and categorical columns
        if continuous_cols is not None and categorical_cols is not None:
            # Fix the column list to not include the target column
            continuous_cols = [col for col in continuous_cols if col in data.columns]
            categorical_cols = [col for col in categorical_cols if col in data.columns]
            # Split the data into continuous and categorical
            data_continuous = data[continuous_cols]
            data_categorical = data[categorical_cols]
            # Compute the mutual information for both types of features
            cont_mi = dict(zip(continuous_cols, mutual_info_regression(data_continuous, y)))
            cat_mi = dict(zip(categorical_cols, mutual_info_classif(data_categorical, y, discrete_features=True)))
            self.mutual_information = pd.Series({**cont_mi, **cat_mi})
        # Otherwise, we calculate the information gain as if all features are continuous
        else:
            mutual_information = dict(zip(data.columns, mutual_info_regression(data, y)))
            self.mutual_information = pd.Series(mutual_information)

    def transform(self, X, num_features=0, min_score=0):
        """
        :param X:
        :param num_features:
        :param min_score:
        :return: The dataframe with only the selected features, the list of selected features and the feature scores
        """
        if num_features == 0 and min_score == 0:
            raise ValueError("Either num_features or min_score must be set")
        # Compute the feature scores series
        feature_scores = self.correlation_weight * self.correlation.values + \
                         (1 - self.correlation_weight) * self.mutual_information
        # Sort the features by their scores
        feature_scores = feature_scores.sort_values(ascending=False)
        possible_features = feature_scores[feature_scores >= min_score]
        if num_features > 0:
            feature_list = possible_features.head(num_features).index
        else:
            feature_list = possible_features.index
        return X[feature_list], feature_list, feature_scores

    def auto_optimize(self, X_train, X_test, y_train, y_test, model: LinearRegression, num_features=0):
        """
        Optimizes the weights for the correlation and information gain, by maximizing the model's performance.
        Optimal weights are found using a binary search.
        :param X:
        :param y:
        :param model:
        :return: a dataframe with the selected features
        """
        best_score = 0
        model_copy = model
        self.correlation_weight = 1
        self.information_gain_weight = 0
