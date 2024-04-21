import numpy as np
import pandas as pd
import os

from deep_network_methods import LinearAgent
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
                                     ['MSSubClass', 'FullBath'], 0.85, target_name='SalePrice')
X, continuous_cols, cat_cols = preprocessor.preprocess()

X_train, X_prod, y_train, y_prod = train_test_split(X, X['SalePrice'], test_size=0.2, random_state=42)
X_train = X_train.drop(columns=['SalePrice'])
X_prod = X_prod.drop(columns=['SalePrice'])

batch_size = 256

model = LinearAgent(X_train, y_train, LinearRegression(), mean_squared_error, batch_size=batch_size, agent_type='A2C',
                    save_path=os.path.join(script_dir, 'models', 'linear_agent'), eval_freq=500)
print(model.agent.policy)
model.learn(num_steps=4000)
model.save(model_name="end_of_training")
# model = LinearAgent.load(os.path.join(script_dir, 'models', 'linear_agent', "best_model.zip"))

X_train_all = X_train.sample(batch_size).to_numpy().astype(np.float32)
action = model.predict(X_train_all.reshape(-1, ))
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



