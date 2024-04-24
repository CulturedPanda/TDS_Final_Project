from .base_method import BaseMethod
import numpy as np
import pandas as pd


class BacktrackingVariant(BaseMethod):
    """
    A variant of the algorithm presented in the paper "A feature selection method based on Shapley values robust to
    concept shift in regression" by Carlos Sebastián and Carlos E. González-Guillén, arXiv preprint arXiv:2304.14774 (2023).
    The original algorithm can be described as a Sequential Feature Selection (SFS) algorithm that removes features
    one by one, always removing the feature with the highest negative influence.
    This variant is a Sequential Floating Feature Selection (SFFS) algorithm that allows for backtracking,
    by checking if adding a feature that was previously removed improves the model.
    This is done because the Shapley values are not monotonic, meaning that a feature that was previously
    removed might be beneficial when added back in a different context.
    """

    __name__ = "ShapValuesBacktrackingVariant"

    def __init__(self, q_low=0.15, q_high=0.85):
        super().__init__(q_low, q_high)

    def _choose_features(self, neg_infs):
        # Remove features with infinite negative influence
        infinite_neg_influence_features = neg_infs[neg_infs == np.infty].index
        neg_infs = neg_infs[neg_infs != np.infty]

        # Find the feature with the highest negative influence
        feature = neg_infs.idxmax()
        feature = pd.Index([feature])

        # Add the feature to remove to the index of infinite negative influence features
        features = feature.append(infinite_neg_influence_features)

        return features

    def predict(self, X_train, y_train, X_val, y_val, model, metric, n_features_to_remove=0,
                metric_lower_is_better=True, num_iter_prev=0):

        if num_iter_prev > 0:
            features = self.preliminary_phase(X_train, y_train, X_val, model.__class__(), num_iter_prev)
            X_train = X_train[features]
            X_val = X_val[features]
        # If n_features to remove is not specified, we simply go until we have no more features to remove
        if n_features_to_remove == 0:
            n_features_to_remove = len(X_train.columns)

        # Initialize a set to keep track of which features have been removed
        removed_features = set()
        # Initialize a closed list, to prevent revisiting the same feature set and getting stuck in a loop
        closed_list = set()
        # Add the original feature set to the closed list, as we don't want to revisit it
        closed_list.add(tuple(X_train.columns))

        # Initialize the best features to be the original features
        features = X_train.columns
        best_features = X_train.columns
        if metric_lower_is_better:
            best_score = np.infty
        else:
            best_score = -np.infty

        while len(features) > 0 and n_features_to_remove > 0:
            current_num_features = len(features)

            # Create a copy of the features to be able to backtrack
            X_train_temp = X_train[features]
            X_val_temp = X_val[features]

            # Find which features should be removed
            features_to_remove = self.main_phase(X_train_temp, y_train, X_val_temp, y_val, model)

            # Add the features to the removed features set
            for feature in features_to_remove:
                if feature not in removed_features:
                    removed_features.add(feature)

            # Remove the features from the current feature set
            X_train_temp = X_train_temp.drop(columns=features_to_remove)
            X_val_temp = X_val_temp.drop(columns=features_to_remove)

            if tuple(X_train_temp.columns) not in closed_list:
                closed_list.add(tuple(X_train_temp.columns))

            # Train a model on the reduced feature set, and evaluate it
            if len(X_train_temp.columns) > 0:
                model = model.__class__()
                model.fit(X_train_temp, y_train)
                y_pred = model.predict(X_val_temp)
                score = metric(y_val, y_pred)
            else:
                score = np.infty if metric_lower_is_better else -np.infty

            # Backtracking step - check if adding a previously removed feature improves the model
            local_best_score = score
            best_local_feature_set = X_train_temp.columns
            current_feature_set = X_train_temp.columns
            for feature in removed_features:
                features = current_feature_set
                features_to_use = features.append(pd.Index([feature]))

                # Make sure we haven't already visited this feature set, as not checking this may lead to infinite loops
                if tuple(features_to_use) in closed_list:
                    continue
                else:
                    closed_list.add(tuple(features_to_use))

                # Train a model on the new feature set, and evaluate it
                X_train_temp = X_train[features_to_use]
                X_val_temp = X_val[features_to_use]
                model = model.__class__()
                model.fit(X_train_temp, y_train)
                y_pred = model.predict(X_val_temp)
                score = metric(y_val, y_pred)

                # If the score is better, update the best feature set and score
                if (metric_lower_is_better and score < local_best_score) or (
                        not metric_lower_is_better and score > local_best_score):
                    local_best_score = score
                    best_local_feature_set = X_train_temp.columns

            # Update the feature set for the next iteration
            features = best_local_feature_set
            n_features_to_remove -= current_num_features - len(features)

            # Update the best feature set and score, if we found a better feature set
            if (metric_lower_is_better and local_best_score < best_score) or (
                    not metric_lower_is_better and local_best_score > best_score):
                best_score = local_best_score
                best_features = best_local_feature_set

        return best_features, best_score
