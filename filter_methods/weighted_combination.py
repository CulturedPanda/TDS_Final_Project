import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression

class WeightedCombination:
    """
    This class implements a filter method, grading the features
    based on a weighted combination of correlation and information gain.
    """

    def __init__(self, correlation_weight=0.5, information_gain_weight=0.5):
        self.information_gain = None
        self.correlation_matrix = None
        self.correlation_weight = correlation_weight
        self.information_gain_weight = information_gain_weight

    def fit(self, X, y):
        self.correlation_matrix = X.corr()
        self.information_gain = mutual_info_classif(X, y)

    def transform(self, X, num_features=0, min_score=0):
        """
        :param X:
        :param num_features:
        :param min_score:
        :return: a dataframe with the selected features
        """
        if num_features == 0 and min_score == 0:
            raise ValueError("Either num_features or min_score must be set")
        feature_scores = self.correlation_weight * self.correlation_matrix.values + \
                         self.information_gain_weight * self.information_gain
        feature_scores = pd.DataFrame(feature_scores, index=X.columns, columns=X.columns)
        feature_scores = feature_scores.stack().reset_index()
        feature_scores.columns = ['Feature1', 'Feature2', 'Score']
        feature_scores = feature_scores[feature_scores['Feature1'] != feature_scores['Feature2']]
        feature_scores = feature_scores[feature_scores['Score'] >= min_score]
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        if num_features > 0:
            feature_scores = feature_scores.head(num_features)
        return feature_scores

    def auto_optimize(self, X_train, X_test, y_train, y_test, model:LinearRegression, num_features=0):
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


