import numpy as np
import pandas as pd
import shap


class BaseMethod:
    """
    An implementation of the method described in the paper "A feature selection method based on Shapley values robust to
    concept shift in regression" by Carlos Sebastián and Carlos E. González-Guillén, arXiv preprint arXiv:2304.14774 (2023).
    """

    __name__ = "ShapValuesBaseMethod"

    def __init__(self, q_low=0, q_high=1):
        """
        :param q_low: the lower quantile of the error distribution
        :param q_high: the upper quantile of the error distribution
        """
        if q_low < 0 or q_low > 1:
            raise ValueError('q_low must be between 0 and 1')
        if q_high < 0 or q_high > 1:
            raise ValueError('q_high must be between 0 and 1')
        if q_low >= q_high:
            raise ValueError('q_low must be less than q_high')
        self.q_low = q_low
        self.q_high = q_high
        self.model = None

    def set_q_low(self, q_low):
        if q_low < 0 or q_low > 1:
            raise ValueError('q_low must be between 0 and 1')
        if q_low >= self.q_high:
            raise ValueError('q_low must be less than q_high')
        self.q_low = q_low

    def set_q_high(self, q_high):
        if q_high < 0 or q_high > 1:
            raise ValueError('q_high must be between 0 and 1')
        if q_high <= self.q_low:
            raise ValueError('q_low must be less than q_high')
        self.q_high = q_high

    def preliminary_phase(self, X_train, y_train, X_val, model, num_iter):
        """
        The preliminary phase of the algorithm, as described in the paper.
        This phase introduces a random variable to the data and computes the global influence of each feature on the
        predictions. Features with less global influence than the random variable are removed.
        This phase is optional, and should be skipped if it removes too many features.
        :param X_train: The training data
        :param y_train: The targets for the training data
        :param X_val: The validation data
        :param model: The model to be used to fit the data
        :param num_iter: The number of iterations for training the model and computing the global influence of each
        feature. Set this to 1 for models without a random component, like linear regression, and to a higher number
        for models with a random component like neural networks. Note: I make no guarantees this code will work with
        neural networks, as I used a specific implementation of SHAP for this method.
        :return: The features to keep
        """
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Compute the SHAP values for the validation data
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_val)

        # Compute the global influence of each feature on the predictions
        global_influence = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_train.columns)

        # Find the most influential feature
        most_influential_feature = global_influence.idxmax()

        # Extract the values of the most influential feature
        most_influential_feature_values_train = X_train[most_influential_feature]
        most_influential_feature_values_val = X_val[most_influential_feature]

        # Create a permutation of the most influential feature values as a random variable
        random_variable_train = np.random.permutation(most_influential_feature_values_train)
        random_variable_val = np.random.permutation(most_influential_feature_values_val)

        # Add the random variable to the training and validation data as a new feature
        X_train_random = X_train.copy()
        X_train_random['random_variable'] = random_variable_train
        X_val_random = X_val.copy()
        X_val_random['random_variable'] = random_variable_val

        # Iterate over the model fitting process num_iter times, and compute the global influence of each feature
        # on the predictions each time.
        # If the model has a random component, this should be done multiple times as the result will vary.
        # Otherwise, for models like linear regression, this can be done once.
        global_influences = []
        for i in range(num_iter):
            model = model.__class__()
            model.fit(X_train_random, y_train)
            explainer = shap.Explainer(model, X_train_random)
            shap_values = explainer(X_val_random)
            global_influence = np.abs(shap_values.values).mean(axis=0)
            global_influence = pd.Series(global_influence, index=X_train_random.columns)
            global_influences.append(global_influence)

        global_influences = pd.DataFrame(global_influences, columns=X_train_random.columns)

        # Compute the mean global influence of each feature
        mean_global_influence = global_influences.mean(axis=0)

        # Remove features with less global influence than the random variable
        features_to_keep = mean_global_influence[
            mean_global_influence >= mean_global_influence['random_variable']].index

        return features_to_keep.drop('random_variable')

    def compute_groups(self, X_train, y_train, X_val, y_val, model):
        """
        Compute the groups of correctly predicted, under-predicted, and over-predicted data points.
        :param X_train: The training data.
        :param y_train: The targets for the training data.
        :param X_val: The validation data.
        :param y_val: The targets for the validation data.
        :param model: The model to be used to fit the data.
        :return: The correctly predicted, under-predicted, and over-predicted data points and the error of the model.
        """
        model.fit(X_train, y_train)
        self.model = model
        y_pred = model.predict(X_val)
        # err(x,y) = y−yˆ(x)
        err = y_val - y_pred

        # Q_low = Quantile(err, q_low) and Q_high the analogue
        Q_low = np.quantile(err, self.q_low)
        Q_high = np.quantile(err, self.q_high)

        #  Let q∗ be the quantile such that P(err ≤ 0) = q∗
        q_star = np.mean(err <= 0)
        # Q∗ the value such that Quantile(err, q∗) = Q∗
        Q_star = np.quantile(err, q_star)

        # Definitions of Q_low* and Q_high*
        if Q_low <= 0 <= Q_high:
            Q_low_star = Q_low
            Q_high_star = Q_high
        elif Q_high < 0:
            Q_low_star = Q_star
            Q_high_star = Q_high - (Q_low - Q_star)
        elif Q_low > 0:
            Q_low_star = Q_low - (Q_high - Q_star)
            Q_high_star = Q_star

        """
        x is correctly predicted if err(x,y) ∈[Q∗_low, Q∗_high].
        x is under predicted if err(x,y) ∈ (Q∗_high, +∞) and over predicted if err(x,y) ∈ (-∞, Q∗_low).
        """
        # The warning of accessing Q_ow_star and Q_high_star before assignment is not a problem because they
        # will always be assigned. I simply did not use Else above to remain consistent with the paper.
        correctly_predicted_indexes = np.where((err >= Q_low_star) & (err <= Q_high_star))[0]
        under_predicted_indexes = np.where(err > Q_high_star)[0]
        over_predicted_indexes = np.where(err < Q_low_star)[0]

        correctly_predicted = X_val.iloc[correctly_predicted_indexes]
        under_predicted = X_val.iloc[under_predicted_indexes]
        over_predicted = X_val.iloc[over_predicted_indexes]

        return correctly_predicted, under_predicted, over_predicted, err

    def _compute_neg_inf(self, columns, ef_cp, ef_up, ef_op, err):
        """
        Computation of negative influence, neg inf_(var), as defined in the paper.
        :param columns: The columns of the data.
        :param ef_cp: An array of the effect of each feature on the correctly predicted group.
        :param ef_up: An array of the effect of each feature on the under-predicted group.
        :param ef_op: An array of the effect of each feature on the over-predicted group.
        :param err: The error of the model.
        :return: A series of the negative influence of each feature.
        """
        neg_infs = []
        q_2_err = np.quantile(err, 0.5)
        for i, column in enumerate(columns):
            # If the feature has no effect on any group, the negative influence is infinite - it contributes nothing
            if np.abs(ef_cp[i]) + np.abs(ef_up[i]) + np.abs(ef_op[i]) == 0:
                neg_inf = np.infty

            # Model is biased towards over predictions and the feature increases the value of over-predictions
            # (undesirable effect), but increases the value of under predictions (desirable effect)
            elif q_2_err < 0 and ef_op[i] > 0 and ef_up[i] > 0 and np.abs(ef_op[i]) > (
                    np.abs(ef_up[i] + np.abs(ef_cp[i]))):
                neg_inf = np.abs(ef_op[i]) - (np.abs(ef_up[i]) + np.abs(ef_cp[i]))

            # Symmetric to the previous case, but for under predictions
            elif q_2_err > 0 and ef_op[i] > 0 and ef_up[i] > 0 and np.abs(ef_up[i]) > (
                    np.abs(ef_op[i] + np.abs(ef_cp[i]))):
                neg_inf = np.abs(ef_up[i]) - (np.abs(ef_op[i]) + np.abs(ef_cp[i]))

            # The feature increases over prediction values and decreases under prediction values - both undesirable
            elif ef_op[i] > 0 and ef_up[i] < 0 and (np.abs(ef_up[i]) + np.abs(ef_op[i])) > np.abs(ef_cp[i]):
                neg_inf = np.abs(ef_up[i]) + np.abs(ef_op[i]) - np.abs(ef_cp[i])

            # Any other case - the feature does not have a negative influence
            else:
                neg_inf = 0
            neg_infs.append(neg_inf)
        neg_infs = pd.Series(neg_infs, index=columns)
        return neg_infs

    def _choose_features(self, neg_infs):
        """
        Choose the features to keep based on the negative influence of the features.
        :param neg_infs: The negative influence of the features.
        :return: The features that should be kept.
        """
        # Remove features with infinite negative influence
        features = neg_infs[neg_infs != np.infty].index
        neg_infs = neg_infs[neg_infs != np.infty]

        # Find the feature with the highest negative influence
        feature = neg_infs.idxmax()
        # Remove the feature with the highest negative influence
        features = features[features != feature]

        # Returns the list of features to be kept
        return features

    def main_phase(self, X_train: pd.DataFrame, y_train, X_val, y_val, model):
        """
        Main phase of the algorithm. Finds which feature(s) to remove by computing which features(s)
        have the most negative influence on the model.
        :param X_train: The training data
        :param y_train: The targets for the training data
        :param X_val: The validation data
        :param y_val: The targets for the validation data
        :param model: The model to be used to fit the data
        :return: A list of features to keep
        """
        correctly_predicted, under_predicted, over_predicted, err = self.compute_groups(X_train, y_train, X_val, y_val,
                                                                                        model)
        explainer = shap.Explainer(self.model, X_val)

        # Compute SHAP values for the correctly predicted, under-predicted, and over-predicted data points
        # Also handle the special case where there are no data points in a group
        if len(correctly_predicted) > 0:
            shap_values_correctly_predicted = explainer(correctly_predicted)
        else:
            shap_values_correctly_predicted = pd.DataFrame(np.zeros((1, len(X_train.columns))))
        if len(under_predicted) > 0:
            shap_values_under_predicted = explainer(under_predicted)
        else:
            shap_values_under_predicted = pd.DataFrame(np.zeros((1, len(X_train.columns))))
        if len(over_predicted) > 0:
            shap_values_over_predicted = explainer(over_predicted)
        else:
            shap_values_over_predicted = pd.DataFrame(np.zeros((1, len(X_train.columns))))

        # Effect_(var,x) = sgn(SHAP_var(x, y, y_hat)) · SHAP_var(x, y, y_hat)^2
        effect_correctly_predicted = np.sign(
            shap_values_correctly_predicted.values) * shap_values_correctly_predicted.values ** 2
        effect_under_predicted = np.sign(shap_values_under_predicted.values) * shap_values_under_predicted.values ** 2
        effect_over_predicted = np.sign(shap_values_over_predicted.values) * shap_values_over_predicted.values ** 2

        # Effect_(var,group) ≡ Ef_(var,group) = Σ_(x ∈ group) Effect_(var,x)
        ef_cp = np.sum(effect_correctly_predicted, axis=0)
        ef_up = np.sum(effect_under_predicted, axis=0)
        ef_op = np.sum(effect_over_predicted, axis=0)

        # Computation of negative influence, neg inf_(var), as defined in the paper
        neg_infs = self._compute_neg_inf(X_train.columns, ef_cp, ef_up, ef_op, err)

        # Returns the list of features to be kept
        return self._choose_features(neg_infs)

    def predict(self, X_train, y_train, X_val, y_val, model, metric, n_features_to_remove=0,
                metric_lower_is_better=True, num_iter_prev=0):
        """
        Predict the best subset of features using the method described in the paper.
        :param X_train: The training data
        :param y_train: The targets for the training data
        :param X_val: The validation data
        :param y_val:  The targets for the validation data
        :param model: The model to be used to fit the data
        :param metric: The metric to be used to evaluate the model
        :param n_features_to_remove: Maximum number of features to remove. If set to 0, the method will continue to
        remove features until there are no more features to remove.
        :param metric_lower_is_better: Determines the comparison operator to be used when comparing the result of the
        metric to the best score. If True, the comparison operator is '<', otherwise it is '>'.
        Set to true for metrics like RMSE, MAE, etc. and False for metrics like R^2, etc.
        :param num_iter_prev: The number of iterations to be used in the preliminary phase. If set to 0, the method will
        only use the main phase. Set to 1 for models without a random component, like linear regression, and to a higher
        number for models with a random component like neural networks. Note: I make no guarantees this code will work
        with neural networks.
        :return: The best subset of features and the best score
        """
        if num_iter_prev > 0:
            features = self.preliminary_phase(X_train, y_train, X_val, model.__class__(), num_iter_prev)
            X_train = X_train[features]
            X_val = X_val[features]
        # If n_features to remove is not specified, we simply go until we have no more features to remove
        if n_features_to_remove == 0:
            n_features_to_remove = len(X_train.columns)
        # Initialize the best features to be the original features
        features = X_train.columns
        best_features = X_train.columns
        if metric_lower_is_better:
            best_score = np.infty
        else:
            best_score = -np.infty
        # So long as we have features to remove and we have not reached the maximum number of features to remove
        while len(features) > 0 and n_features_to_remove > 0:
            current_num_features = len(features)
            # Get the features to keep
            features = self.main_phase(X_train, y_train, X_val, y_val, model.__class__())
            # If we have no more features, break and return the best features
            if len(features) == 0:
                break
            # Keep only the features specified
            X_train = X_train[features]
            X_val = X_val[features]
            # Reset the model
            model = model.__class__()
            # Fit the model and get the prediction
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            # Compute the metric
            score = metric(y_val, y_pred)
            # If the score is better than the best score, update the best score and the best features
            if (metric_lower_is_better and score < best_score) or (not metric_lower_is_better and score > best_score):
                best_score = score
                best_features = features
            # Update the number of features to remove
            n_features_to_remove -= current_num_features - len(features)
        # Return the best subset of features and the best score
        return best_features, best_score

    def auto_optimize(self, X_train, y_train, X_val, y_val, model, metric, n_features_to_remove=0,
                      metric_lower_is_better=True, num_q_low_values=21, min_gap=0.01):
        """
        Find the best q_low and q_high values for the method via grid search.
        :param X_train: The training data.
        :param y_train: The targets for the training data.
        :param X_val: The validation data.
        :param y_val: The targets for the validation data.
        :param model: The model to be used to fit the data.
        :param metric: The metric to be used to evaluate the model.
        :param n_features_to_remove: Maximum number of features to remove. If set to 0, the method will continue to
        remove features until there are no more features to remove.
        :param metric_lower_is_better: Determines the comparison operator to be used when comparing the result of the
        metric to the best score. If True, the comparison operator is '<', otherwise it is '>'.
        :param num_q_low_values: The number of q_low values to be tested. Will be spread evenly between 0 and 1.
        :param min_gap: The minimum gap between q_low and q_high. Must be greater than 0.
        :return: The best q_low and q_high values found, as well as the best subset of features and the best score.
        Please note that the best q_low and q_high values are not guaranteed to be the best for the model, as it is
        impossible to test all possible values. However, they should be close to the best values.
        """
        # Initialize the best values
        best_q_low = 0
        best_q_high = 0
        best_score = np.infty
        best_features = X_train.columns
        # Iterate over the q_low values
        q_low_values = np.linspace(0, 0.99, num_q_low_values)
        for i, q_low in enumerate(q_low_values):
            self.set_q_low(q_low)
            num_q_high_values = num_q_low_values - i
            # Iterate over the q_high values, ensuring that q_high is at least min_gap greater than q_low
            q_high_values = np.linspace(q_low + min_gap, 1, num_q_high_values)
            for q_high in q_high_values:
                self.set_q_high(q_high)
                features, score = self.predict(X_train, y_train, X_val, y_val, model, metric, n_features_to_remove,
                                               metric_lower_is_better)
                print(f"q_low: {q_low}, q_high: {q_high}, score: {score}, num_features: {len(features)}")
                if (metric_lower_is_better and score < best_score) or (
                        not metric_lower_is_better and score > best_score):
                    best_score = score
                    best_features = features
                    best_q_low = q_low
                    best_q_high = q_high
        return best_q_low, best_q_high, best_features, best_score
