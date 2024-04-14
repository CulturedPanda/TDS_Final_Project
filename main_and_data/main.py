import pandas as pd
import os

from filter_methods import WeightedCombination
from data_pre_processing import PreprocessingPipeline

# Get the directory of the currently running script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data file
data_file_path = os.path.join(script_dir, 'data', 'data_houses.csv')

# Read the data file
data = pd.read_csv(data_file_path, index_col="Id")
target = 'SalePrice'
cols = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","FullBath","YearBuilt","YearRemodAdd",
        "LotFrontage","MSSubClass", "SalePrice"]
data = data[cols]
# Select only the first 10 rows
data = data.head(10)

preprocessor = PreprocessingPipeline(data, ['MSSubClass'], 0.9)
X, continuous_cols, cat_cols = preprocessor.preprocess()

filter_method = WeightedCombination()
filter_method.fit(X, target_column="SalePrice", continuous_cols=continuous_cols, categorical_cols=cat_cols)
selected_features = filter_method.transform(X, num_features=5)
print(selected_features)