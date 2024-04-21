import pandas as pd
import os

from shap_values_methods import BranchingVariant, BaseMethod, BacktrackingVariant
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
cols = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd",
        "LotFrontage", "MSSubClass", "SalePrice"]
data = data[cols]

# Preprocess the data to encode categorical variables, impute missing values and remove collinear features
preprocessor = PreprocessingPipeline(data,
                                     ['MSSubClass', 'FullBath'], 0.85, target_name='SalePrice')
X, continuous_cols, cat_cols = preprocessor.preprocess()

X_train, X_test, y_train, y_test = train_test_split(X, X['SalePrice'], test_size=0.10, random_state=42)
X_train, X_prod, y_train, y_prod = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = X_train.drop(columns=['SalePrice'])
X_prod = X_prod.drop(columns=['SalePrice'])
X_test = X_test.drop(columns=['SalePrice'])

model = BacktrackingVariant(q_low=0.15, q_high=0.85)
selected_features, best_score = model.predict(X_train, y_train, X_prod, y_prod, LinearRegression(), mean_squared_error,
                                              num_iter_prev=1)
print(f"Num features: {len(selected_features)}, selected features: {selected_features}, best score: {best_score}")

model = LinearRegression()
model.fit(X_train[selected_features], y_train)
y_pred = model.predict(X_test[selected_features])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred
max_error = residuals.abs().max()
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"With feature selection: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred
max_error = residuals.abs().max()
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Without feature selection: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")

