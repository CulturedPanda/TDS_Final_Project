import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from filter_methods import VarianceInflationFactor, WeightedCombination
from shap_values_methods import BaseMethod, BacktrackingVariant, BranchingVariant
from deep_network_methods import LinearAgent, SequentialAgent
from data_pre_processing import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Get the directory of the currently running script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data file
# data_file_path = os.path.join(script_dir, 'data', 'breast-cancer_logistic.csv')

data_file_path = os.path.join(script_dir, 'data', 'telco_customer_churn_logistic.csv')

data = pd.read_csv(data_file_path)

target = 'Churn'
data.drop(columns=['customerID'], inplace=True)
data[target] = data[target].map({'Yes': 1, 'No': 0})

# target = 'diagnosis'
# data.drop(columns=['id'], inplace=True)
# # Make the target column binary
# data[target] = data[target].map({'M': 1, 'B': 0})
# data = data[cols]

# Preprocess the data to encode categorical variables, impute missing values and remove collinear features
preprocessor = PreprocessingPipeline(data,
                                     cat_cols=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                               'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                               'Contract', 'PaperlessBilling', 'PaymentMethod'],
                                     threshold=0.9, target_name=target)
# preprocessor = PreprocessingPipeline(data, target_name=target, cat_cols=None, threshold=0.9)
X, continuous_cols, cat_cols = preprocessor.preprocess()
scaler = StandardScaler()
X_no_target = X.drop(columns=[target])
X_no_target[continuous_cols] = scaler.fit_transform(X_no_target[continuous_cols])
X = pd.concat([X_no_target, X[target]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, X[target], test_size=0.10, random_state=42)
X_train, X_prod, y_train, y_prod = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=[target])
X_prod = X_prod.drop(columns=[target])
X_test = X_test.drop(columns=[target])

# batch_size = 256
#
# model = LinearAgent(X_train, y_train, LogisticRegression(), log_loss, batch_size=batch_size, agent_type='A2C',
#                     save_path=os.path.join(script_dir, 'models', 'linear_agent_telco'), eval_freq=500)
# print(model.agent.policy)
# model.learn(num_steps=500)
# model.save(model_name="end_of_training")
# # model = LinearAgent.load(os.path.join(script_dir, 'models', 'linear_agent', "best_model.zip"))
#
# X_train_all = X_train.sample(batch_size)
# action = model.predict(X_train_all)
# X = X_prod[X_prod.columns[action == 1]]
# selected_features = X.columns
# print(f"Selected features: {selected_features}")
# print("Num selected features: ", len(selected_features))
#
#
# print(selected_features)
#
# # History - columns are the weights, rows are the number of features, values are the loss
# # Plot the history
# # weights, num_features = np.meshgrid(history.columns, history.index)
# # plt.scatter(weights, num_features, c=history.values, cmap='viridis')
# # plt.xlabel('Weight')
# # plt.ylabel('Number of features')
# # plt.title('Loss vs. Weight and Number of Features')
# # plt.colorbar()
# # plt.show()
#
#
# model = LogisticRegression()
# model.fit(X_train[selected_features], y_train)
# y_pred = model.predict(X_test[selected_features])
# loss = log_loss(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"With feature selection: loss = {loss}, accuracy = {accuracy}, recall = {recall}, precision = {precision}, F1 = {f1}")
#
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# loss = log_loss(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"Without feature selection: loss = {loss}, accuracy = {accuracy}, recall = {recall}, precision = {precision}, F1 = {f1}")


model = SequentialAgent(
    X_train, y_train, 'tenure', LogisticRegression(), log_loss, agent_type='A2C',
    lstm_hidden_layer_size=32, lstm_num_layers=3, network_type='recurrent',
    save_path=os.path.join(script_dir, 'models', 'sequential_model_tesco'),
    eval_freq=50, clustering_method='MeanShift'
)
print(model.agent.policy)
model.learn(num_steps=100)
predictions = model.predict(X_train)
print(predictions)
models, metric_values, predictions, train_sequences, train_targets, test_sequences, test_targets = model.train_models_for_ranges(
    X_train, y_train, X_prod, y_prod, LogisticRegression(),
    [
        log_loss,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    ])
print("Selected features length and columns:")
print(f"Number of sequences: {len(train_sequences)}")
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
    model = LogisticRegression()
    # Select all examples from the original training data that are in the current sequence
    X = X_train.loc[train_sequences[i].index]
    y = y_train.loc[train_sequences[i].index]
    model.fit(X, y)
    X_test = X_prod.loc[test_sequences[i].index]
    y_test = y_prod.loc[test_sequences[i].index]
    y_pred = model.predict(X_test)
    loss = log_loss(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    all_feature_metrics.append({
        'Log loss': loss,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
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
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
loss = log_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(
    f"Without feature selection: loss = {loss}, accuracy = {accuracy}, recall = {recall}, precision = {precision}, F1 = {f1}")
