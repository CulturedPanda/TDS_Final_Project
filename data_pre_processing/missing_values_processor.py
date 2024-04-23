import numpy as np


class MissingValuesProcessor:
    """
    This class is responsible for processing missing values in the data.
    """

    def __init__(self, data):
        self.data = data

    def process(self):
        """
        Fills missing values in the data with the mean of the column.
        :return:
        """
        # Replace completely empty strings with np.NAN, then convert the data to float
        self.data = self.data.replace(r'^\s+$', np.NAN, regex=True)
        self.data = self.data.astype(float)
        # Fill missing values with the mean of the column
        self.data = self.data.fillna(self.data.mean())
        return self.data
