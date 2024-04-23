import pandas as pd


class CategoricalFeatureProcessor:
    """
    This class preprocesses the categorical features in the dataset.
    """
    def __init__(self, df, cat_cols):
        """
        :param df: the dataframe containing the dataset
        :param cat_cols: a list of the names of the categorical columns in the dataset
        """
        self.df = df
        self.cat_cols = cat_cols

    def preprocess(self):
        """
        One-hot encodes the categorical features in the dataset.
        :return: the dataset with one-hot encoded categorical features,
         the names of the continuous columns and the names of the categorical columns
        """
        if self.cat_cols is not None:
            continue_cols = [col for col in self.df.columns if col not in self.cat_cols]
            self.df = pd.get_dummies(self.df, columns=self.cat_cols, drop_first=True)
            cat_cols = [col for col in self.df.columns if col not in continue_cols]
            return self.df, continue_cols, cat_cols
        else:
            continue_cols = self.df.columns
            return self.df, continue_cols, None
