import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VarianceInflationFactor:
    """
    A filter method that uses the variance inflation factor (VIF) to select features.
    VIF is a measure of multicollinearity among the features of a regression model. It quantifies how much the
    variance of the estimated regression coefficients are increased due to multicollinearity.
    Generally, a higher VIF value indicates that the feature is highly collinear with the other features and should be
    removed, and a VIF value of 5 or higher is considered to be high.
    However, due to the nature of some datasets, selecting the features with higher VIF values may lead to better
    performance.
    """

    def __init__(self):
        self.vif = None

    def fit(self, X_train):
        """
        Fit the VIF values to the training data.
        :param X_train: The training data.
        :return:
        """
        self.vif = pd.Series(index=X_train.columns)
        X_train = X_train.astype(float)
        for i, column in enumerate(X_train.columns):
            self.vif[column] = variance_inflation_factor(X_train.values, i)

    def predict(self, X_train, num_features=0, threshold=5, less_than_threshold_comparison=True):
        """
        Predict which features to keep based on the VIF values.
        :param X_train: The training data.
        :param num_features: The maximum number of features to keep. If 0, the number of features is not limited and
        will be equal to the number of features with VIF values above or below the threshold.
        :param threshold: The threshold to use for comparison. If 0, the threshold is not used and the number of
        features is limited by num_features, and the returned features will be the ones with the lowest or highest VIF
        values.
        :param less_than_threshold_comparison: If True, values below the threshold are kept and the sorting direction
        is ascending. If False, values above the threshold are kept and the sorting direction is descending.
        :return: The features to keep.
        """
        if (threshold is None or threshold == 0) and num_features == 0:
            raise ValueError("At least one of threshold and num_features must be different from 0.")
        if num_features == 0:
            num_features = len(X_train)
        if less_than_threshold_comparison:
            features = self.vif[self.vif < threshold]
        else:
            features = self.vif[self.vif > threshold]
        features = features.sort_values(ascending=less_than_threshold_comparison)
        if len(features) >= num_features:
            features = features[:num_features]
        return features.index

    def auto_optimize(self, X_train, y_train, X_test, y_test, model, loss_function):
        """
        This method performs a grid search over the threshold and number of features to select the best combination
        of threshold, number of features and comparison direction that minimizes the loss function.
        :param X_train: The training data.
        :param y_train: The training target.
        :param X_test: The test data.
        :param y_test: The test target.
        :param model: The model to use for training.
        :param loss_function: The loss function to minimize.
        :return: The best features, the best loss, the best threshold, the best number of features, the best comparison
        direction, the history of comparisons with less than comparison direction and the history of comparisons with
        greater than comparison direction.
        """

        # Initialize the grid search space
        threshold_values = np.arange(0.5, 50.5, 0.5)
        num_features_values = np.arange(1, len(X_train.columns) + 1)

        # Initialize the best values
        best_threshold = 0.5
        best_num_features = len(X_train.columns)
        best_loss = np.inf
        best_features = X_train.columns
        best_comparison_direction = None

        # Initialize the history of comparisons
        history_less_than_comparisons = pd.DataFrame(columns=threshold_values, index=num_features_values)
        history_greater_than_comparisons = pd.DataFrame(columns=threshold_values, index=num_features_values)

        # Iterate over the grid search space
        for threshold in threshold_values:
            for num_features in num_features_values:
                for value in [True, False]:

                    # Get the features
                    features = self.predict(X_train, num_features, threshold, value)

                    # Reset the model, train it and get the loss value
                    if len(features) > 0:
                        model = model.__class__()
                        model.fit(X_train[features], y_train)
                        y_pred = model.predict(X_test[features])
                        loss = loss_function(y_test, y_pred)
                    else:
                        loss = np.inf

                    # Update the best values if necessary. Priority is given to lower loss, then to either
                    # lower number of features or higher threshold
                    if ((loss < best_loss)
                            or (loss == best_loss and num_features < best_num_features)
                            or (loss == best_loss and best_threshold < threshold)):
                        best_loss = loss
                        best_threshold = threshold
                        best_num_features = num_features
                        best_features = features
                        best_comparison_direction = value

                    # Update the history
                    if value:
                        history_less_than_comparisons.loc[num_features, threshold] = loss
                    else:
                        history_greater_than_comparisons.loc[num_features, threshold] = loss

                    print(f"Threshold: {threshold}, Num features: {num_features}, Less than comparison: {value}, Loss: {loss}")

        print(f"Best threshold: {best_threshold}, Best num features: {best_num_features}, Best loss: {best_loss}, "
              f"Best comparison direction: {'less than' if best_comparison_direction else 'greater than'}")

        return (best_features, best_loss, best_threshold, best_num_features, best_comparison_direction,
                history_less_than_comparisons, history_greater_than_comparisons)
