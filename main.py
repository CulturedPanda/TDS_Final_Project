import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from filter_methods import VarianceInflationFactor, WeightedCombination
from shap_values_methods import BaseMethod, BacktrackingVariant, BranchingVariant
from data_pre_processing import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/Life Expectancy Data.csv")
data = data.drop(columns=['Country'])
# data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
num_rows = data.shape[0]
num_cols = data.shape[1]
print(f"Number of rows: {num_rows}, number of columns: {num_cols}")

target = 'Life expectancy '

# target = 'diagnosis'
# data.drop(columns=['id'], inplace=True)
# # Make the target column binary
# data[target] = data[target].map({'M': 1, 'B': 0})
# data = data[cols]

# Preprocess the data to encode categorical variables, impute missing values and remove collinear features
preprocessor = PreprocessingPipeline(data, cat_cols=['Status'],
                                     threshold=0.9, target_name=target)
X, continuous_cols, cat_cols = preprocessor.preprocess()
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

X_train, X_test, y_train, y_test = train_test_split(X, X[target], test_size=0.10, random_state=42)
X_train, X_prod, y_train, y_prod = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=[target])
X_prod = X_prod.drop(columns=[target])
X_test = X_test.drop(columns=[target])

selector = BacktrackingVariant(q_low=0.15, q_high=0.85)
selected_features, best_score = selector.predict(X_train, y_train, X_prod, y_prod, LogisticRegression(), log_loss, num_iter_prev=1)
print(f"Selected features: {selected_features}, best score: {best_score}")
print(f"Number of selected features: {len(selected_features)}")

# History - columns are the weights, rows are the number of features, values are the loss
# Plot the history
# weights, num_features = np.meshgrid(history.columns, history.index)
# plt.scatter(weights, num_features, c=history.values, cmap='viridis')
# plt.xlabel('Weight')
# plt.ylabel('Number of features')
# plt.title('Loss vs. Weight and Number of Features')
# plt.colorbar()
# plt.show()

model = LogisticRegression()
model.fit(X_train[selected_features], y_train)
y_pred = model.predict(X_test[selected_features])
loss = log_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(
    f"With feature selection: loss = {loss}, accuracy = {accuracy}, recall = {recall}, precision = {precision}, F1 = {f1}")

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
loss = log_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(
    f"Without feature selection: loss = {loss}, accuracy = {accuracy}, recall = {recall}, precision = {precision}, F1 = {f1}")
