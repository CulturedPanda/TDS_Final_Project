import numpy as np
import pandas as pd
import os

from filter_methods import WeightedCombination
from data_pre_processing import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the data file
data = pd.read_csv('data/insurance.csv')
target = 'charges'
print(f"Num rows: {data.shape[0]}, num cols: {data.shape[1]}")

# Preprocess the data to encode categorical variables, impute missing values and remove collinear features
preprocessor = PreprocessingPipeline(data, ['sex', 'smoker', 'region'], 0.9, target_name='SalePrice')
X, continuous_cols, cat_cols = preprocessor.preprocess()

X_train, X_test, y_train, y_test = train_test_split(X, X[target], test_size=0.10, random_state=42)
X_train, X_prod, y_train, y_prod = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

filter_method = WeightedCombination()
print(filter_method.__name__)
filter_method.fit(X_train, target_column=target, continuous_cols=continuous_cols, categorical_cols=cat_cols)
X_transformed, selected_features, feature_scores, best_weight, best_loss, best_num_features, best_threshold, history \
    = filter_method.auto_optimize(X_train,
                                  y_train, X_prod,
                                  y_prod,
                                  loss_function=mean_squared_error)
# Use the best weight found during optimization
# filter_method.set_weight(0.41)
# X_transformed, selected_features, _ = filter_method.transform(X_train, num_features=13)
print(selected_features)

# History - columns are the weights, rows are the number of features, values are the loss
# Plot the history
weights, num_features = np.meshgrid(history.columns, history.index)
plt.scatter(weights, num_features, c=history.values, cmap='viridis')
plt.xlabel('Weight')
plt.ylabel('Number of features')
plt.title('Loss vs. Weight and Number of Features')
plt.colorbar()
plt.show()


X_train = X_train.drop(columns=[target])
X_test = X_test.drop(columns=[target])
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
