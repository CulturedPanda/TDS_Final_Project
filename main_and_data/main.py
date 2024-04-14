import pandas as pd
import os

from deep_network_methods import FullyConnectedSelector
from data_pre_processing import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

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

# Preprocess the data to encode categorical variables, impute missing values and remove collinear features
preprocessor = PreprocessingPipeline(data,
                                     ['MSSubClass', 'FullBath'], 0.85)
X, continuous_cols, cat_cols = preprocessor.preprocess()

X_train, X_prod, y_train, y_prod = train_test_split(X.drop("SalePrice"), X['SalePrice'], test_size=0.2, random_state=42)

selection_method = FullyConnectedSelector(X_train, y_train, LinearRegression(), mean_squared_error)
selection_method.train(1000)


