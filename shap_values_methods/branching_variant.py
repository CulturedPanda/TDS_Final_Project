from .base_method import BaseMethod
import numpy as np
import pandas as pd
from queue import Queue


class BranchingVariant(BaseMethod):
    """
    A variant of the algorithm presented in the paper "A feature selection method based on Shapley values robust to
    concept shift in regression" by Carlos Sebastián and Carlos E. González-Guillén, arXiv preprint arXiv:2304.14774 (2023).
    While the original algorithm makes the greedy choice of always removing the feature with the highest negative
    influence, this variant creates branches, where one branch removes the feature with the highest negative influence
    and the other removes the feature with the second-highest negative influence.
    This allows the algorithm to explore more possibilities and potentially find better subsets of features.
    """

    def __init__(self, q_low=0.15, q_high=0.85):
        super().__init__(q_low, q_high)

    def _choose_features(self, neg_infs):
        # Removes features with infinite negative influence
        features = neg_infs[neg_infs != np.infty].index
        neg_infs = neg_infs[neg_infs != np.infty]

        # Choose the feature with the highest and second-highest negative influence
        neg_infs = neg_infs.sort_values(ascending=False)
        best_choices = neg_infs.index[:2]

        best_choice = best_choices[0]
        if len(best_choices) > 1:
            second_choice = best_choices[1]
        else:
            second_choice = best_choice

        # Returns the list of features to be kept
        return features[features != best_choice], features[features != second_choice]

    def predict(self, X_train, y_train, X_val, y_val, model, metric, n_features_to_remove=0,
                metric_lower_is_better=True, num_iter_prev=0):

        if num_iter_prev > 0:
            features = self.preliminary_phase(X_train, y_train, X_val, model.__class__(), num_iter_prev)
            X_train = X_train[features]
            X_val = X_val[features]
        # If n_features to remove is not specified, we simply go until we have no more features to remove
        if n_features_to_remove == 0:
            n_features_to_remove = len(X_train.columns)

        # Initialize the open and closed lists
        open_list = Queue()
        open_list.put(tuple(X_train.columns))
        closed_list = set()
        closed_list.add(tuple(X_train.columns))

        # Initialize the best features to be the original features
        total_features = len(X_train.columns)
        best_features = X_train.columns
        if metric_lower_is_better:
            best_score = np.infty
        else:
            best_score = -np.infty

        # So long as we can still find options for features, we continue
        while not open_list.empty():
            # Get the features to be tested
            features = pd.Index(open_list.get())
            # Select only those features
            X_train_temp = X_train[features]
            X_val_temp = X_val[features]
            # Let the main phase of the algorithm choose the best and second best features to remove
            best_choice, second_best_choice = self.main_phase(X_train_temp, y_train, X_val_temp, y_val, model)
            # Tuples are hashable while pd indexes are not
            best_choice = tuple(best_choice)
            second_best_choice = tuple(second_best_choice)
            # If we have not added the subsets of features to the closed list, we add them to the open list (so long
            # as they are not empty)
            if best_choice not in closed_list and len(best_choice) >= 1 and len(best_choice) >= (
                    total_features - n_features_to_remove):
                closed_list.add(best_choice)
                open_list.put(best_choice)
            if second_best_choice not in closed_list and len(second_best_choice) >= 1 and len(second_best_choice) >= (
                    total_features - n_features_to_remove):
                closed_list.add(second_best_choice)
                open_list.put(second_best_choice)

        # Compute the best score and features, over the subsets of features we have found
        for subset in closed_list:
            model = model.__class__()
            subset = pd.Index(subset)
            X_train_temp = X_train[subset]
            X_val_temp = X_val[subset]
            model.fit(X_train_temp, y_train)
            y_pred = model.predict(X_val_temp)
            score = metric(y_val, y_pred)
            if (metric_lower_is_better and score < best_score) or (not metric_lower_is_better and score > best_score):
                best_score = score
                best_features = subset

        return best_features, best_score
