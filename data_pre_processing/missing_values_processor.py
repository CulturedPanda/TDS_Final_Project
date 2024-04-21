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
        self.data = self.data.fillna(self.data.mean())
        return self.data