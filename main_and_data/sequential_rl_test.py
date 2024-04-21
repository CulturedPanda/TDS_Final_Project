import numpy as np
import pandas as pd
import os

from deep_network_methods import SequentialAgent
from data_pre_processing import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, max_error
from sklearn.linear_model import LinearRegression
from stable_baselines3.common.env_checker import check_env

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
                                     ['MSSubClass', 'FullBath'], 0.85)
X, continuous_cols, cat_cols = preprocessor.preprocess()

X_train, X_prod, y_train, y_prod = train_test_split(X, X['SalePrice'], test_size=0.2, random_state=42)
X_train = X_train.drop(columns=['SalePrice'])
X_prod = X_prod.drop(columns=['SalePrice'])

batch_size = 256

model = SequentialAgent(
    X_train, y_train, 'YearBuilt', LinearRegression(), mean_squared_error, agent_type='A2C',
    lstm_hidden_layer_size=32, lstm_num_layers=3, network_type='recurrent',
    save_path=os.path.join(script_dir, 'models', 'sequential_model')
)
print(model.agent.policy)
model.learn(num_steps=2000)
predictions = model.predict(X_train)
print(predictions)
models, metric_values, predictions, train_sequences, train_targets, test_sequences, test_targets = model.train_models_for_ranges(
    X_train, y_train, X_prod, y_prod, LinearRegression(),
    [
        mean_squared_error,
        r2_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        max_error
    ])
print("Selected features length and columns:")
print([len(prediction) for prediction in predictions])
print([sequence.columns for sequence in train_sequences])
print("\n")
print(metric_values)
average_metrics = {metric: np.mean([model_metrics[metric] for model_metrics in metric_values]) for metric in
                   metric_values[0].keys()}
max_metrics = {metric: np.max([model_metrics[metric] for model_metrics in metric_values]) for metric in
                metric_values[0].keys()}
min_metrics = {metric: np.min([model_metrics[metric] for model_metrics in metric_values]) for metric in
                metric_values[0].keys()}
print("With feature selection, with sequencing:")
print(f"Average: {average_metrics}")
print(f"Max: {max_metrics}")
print(f"Min: {min_metrics}")
print("\n")

all_feature_metrics = []

print("Without feature selection, with sequencing:")
for i in range(len(train_sequences)):
    model = LinearRegression()
    # Select all examples from the original training data that are in the current sequence
    X = X_train.loc[train_sequences[i].index]
    y = y_train.loc[train_sequences[i].index]
    model.fit(X, y)
    X_test = X_prod.loc[test_sequences[i].index]
    y_test = y_prod.loc[test_sequences[i].index]
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    residuals = y_test - y_pred
    max_error = residuals.abs().max()
    all_feature_metrics.append({
        'MSE': mse,
        'R2': r2,
        'MAE': mae,
        'MAPE': mape,
        'Max error': max_error
    })

average_metrics = {metric: np.mean([model_metrics[metric] for model_metrics in all_feature_metrics]) for metric in
                   all_feature_metrics[0].keys()}
max_metrics = {metric: np.max([model_metrics[metric] for model_metrics in all_feature_metrics]) for metric in
                all_feature_metrics[0].keys()}
min_metrics = {metric: np.min([model_metrics[metric] for model_metrics in all_feature_metrics]) for metric in
                all_feature_metrics[0].keys()}
print(f"Average: {average_metrics}")
print(f"Max: {max_metrics}")
print(f"Min: {min_metrics}")
print("\n")

print("Baseline model - no feature selection, no sequencing:")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_prod)
mse = mean_squared_error(y_prod, y_pred)
r2 = r2_score(y_prod, y_pred)
mae = mean_absolute_error(y_prod, y_pred)
mape = mean_absolute_percentage_error(y_prod, y_pred)
residuals = y_prod - y_pred
max_error = residuals.abs().max()
print(f"Without feature selection, without sequencing: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")




# X_train_all = X_train.sample(batch_size).to_numpy().astype(np.float32)
# action, _states = model.predict(X_train_all.reshape(-1, ))
# X = X_prod[X_prod.columns[action == 1]]
# selected_features = X.columns
# print(f"Selected features: {selected_features}")
# print("Num selected features: ", len(selected_features))
#
# lin_model = LinearRegression()
# lin_model.fit(X_train[selected_features], y_train)
# y_pred = lin_model.predict(X_prod[selected_features])
# mse = mean_squared_error(y_prod, y_pred)
# r2 = r2_score(y_prod, y_pred)
# mae = mean_absolute_error(y_prod, y_pred)
# mape = mean_absolute_percentage_error(y_prod, y_pred)
# residuals = y_prod - y_pred
# max_error = residuals.abs().max()
# print(f"With feature selection: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")
#
# lin_model = LinearRegression()
# lin_model.fit(X_train, y_train)
# y_pred = lin_model.predict(X_prod)
# mse = mean_squared_error(y_prod, y_pred)
# r2 = r2_score(y_prod, y_pred)
# mae = mean_absolute_error(y_prod, y_pred)
# mape = mean_absolute_percentage_error(y_prod, y_pred)
# residuals = y_prod - y_pred
# max_error = residuals.abs().max()
# print(f"Without feature selection: MSE = {mse}, R2 = {r2}, max error = {max_error}, MAE = {mae}, MAPE = {mape}")
