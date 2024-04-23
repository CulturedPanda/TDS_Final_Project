import pandas as pd


def is_boolean_column(series: pd.Series):
    """
    Check if a column is a boolean column.
    :param pd.Series: The column to check.
    :return: True if the column is a boolean column, False otherwise.
    """
    return series.isin([0, 1]).all()