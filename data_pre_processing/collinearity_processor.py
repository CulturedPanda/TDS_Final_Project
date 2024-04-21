class CollinearityProcessor:
    """
    A processor that removes collinear features from a dataset.
    Will remove one of the collinear features, but not both.
    The remaining feature is the one that has the highest correlation with the target.
    The threshold parameter determines the minimum correlation between two features to be considered collinear.
    """

    def __init__(self, target_name, threshold=0.85):
        """

        :param target_name: The name of the target column
        :param threshold: The minimum correlation between two features to be considered collinear
        """
        self.correlation_matrix = None
        self.threshold = threshold
        self.target_name = target_name

    def fit(self, X):
        """
        Fits the processor to the dataset.
        :param X: The dataset
        :return:
        """
        self.correlation_matrix = X.corr()

    def transform(self, X):
        """
        Removes the collinear features from the dataset.
        :param X: The dataset.
        :return: The dataset with the collinear features removed
        """
        return X.drop(columns=self._get_collinear_features())

    def _get_collinear_features(self):
        """
        Finds the collinear features in the dataset.
        :param X: The dataset.
        :return: A set of the collinear features
        """
        collinear_features = set()
        # For each pair of features, if the correlation between them is greater than the threshold, add one of them to
        # the set
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i):
                if abs(self.correlation_matrix.iloc[i, j]) > self.threshold:
                    collinear_features.add(self._get_feature_to_remove(i, j))
        return collinear_features

    def _get_feature_to_remove(self, i, j):
        """
        Determines which of two collinear features to remove.
        :param i: The index of the first feature.
        :param j: The index of the second feature.
        :return:
        """
        feature_i = self.correlation_matrix.columns[i]
        feature_j = self.correlation_matrix.columns[j]
        if (abs(self.correlation_matrix.loc[feature_i, self.target_name])
                > abs(self.correlation_matrix.loc[feature_j, self.target_name])):
            return feature_j
        return feature_i
