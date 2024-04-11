import pandas as pd


class CategoricalFeatureProcessor:
    """
    This class preprocesses the categorical features in the dataset.
    """
    def __init__(self, df, cat_cols):
        self.df = df
        self.cat_cols = cat_cols

    def preprocess(self):
        self.df = pd.get_dummies(self.df, columns=self.cat_cols, drop_first=True)
        return self.df
