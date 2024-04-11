class CollinearityProcessor:
    """
    A processor that removes collinear features from a dataset.
    Will remove one of the collinear features, but not both.
    The remaining feature is the one that has the highest correlation with the target.
    The threshold parameter determines the minimum correlation between two features to be considered collinear.
    """
    def __init__(self, threshold=0.85):
        self.correlation_matrix = None
        self.threshold = threshold

    def fit(self, X):
        self.correlation_matrix = X.corr()

    def transform(self, X):
        return X.drop(columns=self._get_collinear_features(X))

    def _get_collinear_features(self, X):
        collinear_features = set()
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i):
                if abs(self.correlation_matrix.iloc[i, j]) > self.threshold:
                    collinear_features.add(self._get_feature_to_remove(i, j, X))
        return collinear_features

    def _get_feature_to_remove(self, i, j, X):
        feature_i = self.correlation_matrix.columns[i]
        feature_j = self.correlation_matrix.columns[j]
        target = X.columns[-1]
        if abs(self.correlation_matrix.loc[feature_i, target]) > abs(self.correlation_matrix.loc[feature_j, target]):
            return feature_j
        return feature_i