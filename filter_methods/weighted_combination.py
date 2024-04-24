import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error


class WeightedCombination:
    """
    This class implements a filter method, grading the features
    based on a weighted combination of correlation and information gain.
    Also provides a method to optimize the weights for the correlation and information gain as well as the number of
    features to select.
    """

    __name__ = "WeightedCombination"

    def __init__(self, correlation_weight=0.5, target_column=None):
        self.mutual_information = None
        self.correlation = None
        self.correlation_weight = correlation_weight
        self.target_column = target_column

    def set_weight(self, correlation_weight):
        self.correlation_weight = correlation_weight

    def fit(self, X, target_column=None, continuous_cols=None, categorical_cols=None):
        """
        Fits the filter method to the data.
        :param X: The data to fit the filter method to.
        :param target_column: The name of the target column.
        :param continuous_cols: A list of the names of the continuous columns in the data.
        :param categorical_cols: A list of the names of the categorical columns in the data.
        :return: None
        """
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
            if len(continuous_cols) > 0:
                cont_mi = dict(zip(continuous_cols, mutual_info_regression(data_continuous, y)))
            else:
                cont_mi = {}
            if len(categorical_cols) > 0:
                # We set y to int, as the mutual_info_classif function requires the target to be an integer
                cat_mi = dict(zip(categorical_cols, mutual_info_classif(data_categorical, y.astype(int), discrete_features=True)))
            else:
                cat_mi = {}
            self.mutual_information = pd.Series({**cont_mi, **cat_mi})

        # Otherwise, we calculate the information gain as if all features are continuous
        else:
            mutual_information = dict(zip(data.columns, mutual_info_regression(data, y)))
            self.mutual_information = pd.Series(mutual_information)

    def transform(self, X, num_features=0, min_threshold=0):
        """
        Transforms the data to only include the selected features.
        Either the number of features or the minimum threshold must be set.
        :param X: The data to transform.
        :param num_features: The number of features to select. Default is 0, which means all features with a score above
        the threshold will be selected.
        :param min_threshold: The minimum score a feature must have to be selected. Default is 0, which means all
        features with a score above the threshold will be selected.
        :return: The dataframe with only the selected features, the list of selected features and the feature scores.
        """
        if num_features == 0 and min_threshold == 0:
            raise ValueError("Either num_features or min_threshold must be set")

        # Compute the feature scores series
        feature_scores = self.correlation_weight * self.correlation.values + \
                         (1 - self.correlation_weight) * self.mutual_information

        # Sort the features by their scores
        feature_scores = feature_scores.sort_values(ascending=False)

        # Get the features with a score above the threshold
        possible_features = feature_scores[feature_scores >= min_threshold]

        if num_features > 0:
            try:
                feature_list = possible_features.head(num_features).index
            # If the number of features is greater than the number of features with a score above the threshold
            # we will select all the features with a score above the threshold
            except ValueError:
                feature_list = possible_features.index
        else:
            feature_list = possible_features.index
        return X[feature_list], feature_list, feature_scores

    def _test_on_values(self, X_train, y_train, X_test, y_test,
                        model, num_features, parameter_value, loss_function, threshold=0):
        """
        Tests the performance of the model on the selected features.
        :param X_train: The training data.
        :param y_train: The training target.
        :param X_test: The test data.
        :param y_test: The test target.
        :param model: The model to be used for the optimization.
        :return: the model's performance
        """

        # Copy the uninitialized model
        model = model.__class__()

        # Set the correlation weight to the parameter value
        self.correlation_weight = parameter_value

        # Get the selected features
        X_train, selected_features, _ = self.transform(X_train, num_features=num_features, min_threshold=threshold)

        # If no features were selected, return infinity as the loss
        if len(selected_features) == 0:
            return np.inf

        X_test = X_test[selected_features]

        # Fit the model and get the predictions
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Return the model's performance
        return loss_function(y_test, preds)

    def auto_optimize(self, X_train, y_train, X_test, y_test,
                      model: sklearn.linear_model = sklearn.linear_model.LinearRegression(),
                      loss_function: callable = mean_squared_error,
                      include_threshold_optimization=False):
        """
        Optimizes the weights for the correlation and information gain, by maximizing the model's performance.
        Optimal weights are found using a grid search on the correlation weight and number of features. :param X:
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param model: A model to be used for the optimization. Must have a fit and predict method. Default
        is LinearRegression. For custom models, make sure the model class has the fit function,
        with pre-defined hyperparameters.
        :param loss_function: A loss function to be used for the optimization. Default is
        mean_squared_error.
        :param include_threshold_optimization: A boolean indicating whether to include the optimization of the
        min_threshold parameter. Default is False. Note that this will significantly increase the time taken,
        as it will perform a 3d grid search instead of a 2d grid search.
        :return: a dataframe with the selected features, the list of selected features and the feature scores,
        the best weight, the best loss, the best number of features, the best threshold found,
        and a dataframe with the optimization history, where the columns are the weights, the rows are the number of features,
        and the values are the losses.
        """

        # Initialize the best loss and weight
        best_loss = np.inf
        best_weight = None
        best_num_features = None
        best_threshold = 0


        # Initialize the grid search parameters
        correlation_weights = np.linspace(0, 1, 101)
        num_features = np.arange(1, len(X_train.columns) + 1)
        threshold_weights = np.linspace(0, 1, 101)

        # Initialize the optimization history - a dataframe where columns are the weight, row is the number of features,
        # and the value is the loss
        history = pd.DataFrame(columns=correlation_weights, index=num_features)

        # Iterate over the grid search parameters.
        # If include_threshold_optimization is False, we will skip the threshold optimization
        if not include_threshold_optimization:
            for weight in correlation_weights:
                for num in num_features:
                    loss = self._test_on_values(X_train, y_train, X_test, y_test, model, num, weight, loss_function)
                    print(f"Weight: {weight}, Num features: {num}, Loss: {loss}")

                    # Update the best loss, weight and number of features if the current loss is better than the best
                    # loss or if the current loss is equal to the best loss and the current number of features is less
                    # than the best number of features
                    if (loss < best_loss) or (loss == best_loss and num < best_num_features):
                        best_loss = loss
                        best_weight = weight
                        best_num_features = num
                    history.loc[num, weight] = loss
        # If include_threshold_optimization is True, we will include the threshold optimization
        else:
            for weight in correlation_weights:
                for num in num_features:
                    for threshold_weight in threshold_weights:
                        loss = self._test_on_values(X_train, y_train, X_test, y_test, model, num, weight, loss_function,
                                                    threshold=threshold_weight)
                        if loss == np.inf:
                            print("No features selected, skipping")
                            break
                        print(
                            f"Weight: {weight}, Num features: {num}, Score threshold: {threshold_weight}, Loss: {loss}")

                        # Update the best loss, weight, number of features and threshold if the current loss is better
                        if (loss < best_loss) or (loss == best_loss and num < best_num_features):
                            best_loss = loss
                            best_weight = weight
                            best_num_features = num
                            best_threshold = threshold_weight

        # Set the best weight and number of features
        self.correlation_weight = best_weight

        # Get the selected features
        print(f"Best weight: {best_weight}, Best loss: {best_loss}, Best num features: {best_num_features}")
        X_transformed, selected_features, feature_scores = self.transform(X_train, num_features=best_num_features)
        history = history.astype(float)
        return (X_transformed, selected_features, feature_scores, best_weight, best_loss, best_num_features,
                best_threshold, history)
