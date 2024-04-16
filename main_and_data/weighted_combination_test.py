import pandas as pd
import os

from filter_methods import WeightedCombination
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

X_train, X_test, y_train, y_test = train_test_split(X, X['SalePrice'], test_size=0.10, random_state=42)
X_train, X_prod, y_train, y_prod = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

filter_method = WeightedCombination()
filter_method.fit(X_train, target_column="SalePrice", continuous_cols=continuous_cols, categorical_cols=cat_cols)
# X_transformed, selected_features, feature_scores, best_weight, best_loss, best_num_features, best_threshold \
#     = filter_method.auto_optimize(X_train,
#                                   y_train, X_prod,
#                                   y_prod,
#                                   loss_function=mean_squared_error)
# Use the best weight found during optimization
filter_method.set_weight(0.41)
X_transformed, selected_features, _ = filter_method.transform(X_train, num_features=13)
print(selected_features)

X_train = X_train.drop(columns=['SalePrice'])
X_test = X_test.drop(columns=['SalePrice'])
model = LinearRegression()
model.fit(X_transformed, y_train)
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
