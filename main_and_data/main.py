import numpy as np
import pandas as pd
import os

from deep_network_methods import DatasetEnv
from data_pre_processing import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

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

X_train, X_prod, y_train, y_prod = train_test_split(X, X['SalePrice'], test_size=0.2, random_state=42)
X_train = X_train.drop(columns=['SalePrice'])
X_prod = X_prod.drop(columns=['SalePrice'])

environment = DatasetEnv(X_train, y_train, X_prod, y_prod, LinearRegression(), mean_squared_error, batch_size=64)
check_env(environment)
model = A2C("MlpPolicy", environment, verbose=1).learn(4000)
mean_reward, std_reward = evaluate_policy(model, environment, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

X_prod_64 = X_prod.sample(64)
y_prod_64 = y_prod.loc[X_prod_64.index]
X_prod_64 = X_prod_64.to_numpy().astype(np.dtype('float32')).reshape(-1, )
action, _states = model.predict(X_prod_64, deterministic=True)
X = X_prod[X_prod.columns[action == 1]]
selected_features = X.columns
print(f"Selected features: {selected_features}")
print("Num selected features: ", len(selected_features))

lin_model = LinearRegression()
lin_model.fit(X_train[selected_features], y_train)
y_pred = lin_model.predict(X_prod[selected_features])
mse = mean_squared_error(y_prod, y_pred)
r2 = r2_score(y_prod, y_pred)
mae = mean_absolute_error(y_prod, y_pred)
mape = mean_absolute_percentage_error(y_prod, y_pred)
residuals = y_prod - y_pred
max_error = residuals.abs().max()
print(f"With feature selection: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_prod)
mse = mean_squared_error(y_prod, y_pred)
r2 = r2_score(y_prod, y_pred)
mae = mean_absolute_error(y_prod, y_pred)
mape = mean_absolute_percentage_error(y_prod, y_pred)
residuals = y_prod - y_pred
max_error = residuals.abs().max()
print(f"Without feature selection: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")



