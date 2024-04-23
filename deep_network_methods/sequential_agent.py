import numpy as np
import os

import pandas as pd

from .sequential_model_dataset_env import SequentialModelDatasetEnv
from .default_fully_connected_network import DefaultFullyConnectedNetwork
from .default_sequence_modeling_network import DefaultSequenceModelingNetwork
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback


class SequentialAgent:
    """
    An agent that treats the data as sequences and learns to select features from the sequences.
    Can use either a fully connected network or a recurrent network.
    """

    def __init__(self, X_train, y_train, col: str, downstream_model, loss_function: callable,
                 agent_type: str = 'A2C', network_type='recurrent',
                 network_architecture_class: BaseFeaturesExtractor | str = None,
                 environment: SequentialModelDatasetEnv = None, network_kwargs: dict = None,
                 save_path: str = None, eval_freq: int = 250, lstm_hidden_layer_size: int = 128,
                 lstm_num_layers: int = 3, clustering_method='MeanShift'):
        """
        Initialize the agent.
        :param X_train: The training data.
        :param y_train: The training targets.
        :param col: The column to sequence by.
        :param downstream_model: The downstream model that will be used for prediction on the data, after feature selection.
        :param loss_function: The loss function to use for the downstream model.
        :param agent_type: The type of agent to use. Can be 'A2C', 'PPO' or 'RecurrentPPO'. Default is 'A2C'. Note that
        'RecurrentPPO' only works with default arguments, while 'A2C' and 'PPO' work with custom arguments.
        :param network_type: The type of network to use. Can be 'linear' or 'recurrent'. Default is 'recurrent'.
        :param network_architecture_class: The class of the network architecture to use. If None, a default network
        architecture of the type specified by network_type is used. If 'auto', the network architecture is determined
        automatically by the Stable Baselines3 library's defaults.
        :param environment: A gymnasium environment to use. If None, a default environment is created.
        The environment is used to sequence the data and train the downstream model on the sequences.
        It must have the following methods: step, reset, get_sequencer
        :param network_kwargs: Arguments to pass to the network architecture class. If None, default arguments are used.
        :param save_path: The path to save the best model to.
        :param eval_freq: The frequency at which to evaluate the model.
        """

        # For all policies except RecurrentPPO, the data must be flattened. This is due to design choices in the
        # DefaultFullyConnectedNetwork and DefaultSequenceModelingNetwork classes.
        self.flatten = (agent_type != 'RecurrentPPO')
        self.feature_space_dim = len(X_train.columns)
        self.data_columns = X_train.columns

        # Initialize the environment if none is provided
        if environment is None:
            self.environment = SequentialModelDatasetEnv(X_train=X_train,
                                                         y_train=y_train,
                                                         col=col,
                                                         downstream_model=downstream_model,
                                                         loss_function=loss_function,
                                                         flatten=self.flatten,
                                                         clustering_method=clustering_method)
        else:
            self.environment = environment
        self.agent_type = agent_type

        # If no network architecture is specified, use the default network architecture of the type specified by
        # network_type
        if network_architecture_class is None:
            if network_type == 'linear':
                self.network_architecture_class = DefaultFullyConnectedNetwork
                network_kwargs = dict(
                    input_size=self.environment.observation_space.shape[0],
                    features_dim=X_train.shape[1],
                )
            elif network_type == 'recurrent':
                self.network_architecture_class = DefaultSequenceModelingNetwork
                network_kwargs = dict(
                    features_dim=X_train.shape[1],
                    hidden_size=lstm_hidden_layer_size,
                    num_layers=lstm_num_layers
                )

        # If the network architecture is set to 'auto', use the default network architecture of the agent
        else:
            self.network_architecture_class = network_architecture_class
        if self.network_architecture_class == "auto":
            policy_kwargs = None
        else:
            policy_kwargs = dict(
                features_extractor_class=self.network_architecture_class,
                features_extractor_kwargs=network_kwargs,
            )
        self.save_path = save_path
        self.eval_callback = EvalCallback(self.environment, best_model_save_path=save_path, log_path=save_path,
                                          eval_freq=eval_freq, deterministic=True)

        # Create the agent
        if agent_type == 'A2C':
            self.agent = A2C("MlpPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)
        elif agent_type == 'PPO':
            self.agent = PPO("MlpPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)
        # Recurrent PPO agent only works with default arguments, because I can not, for the life of me,
        # get it to work with custom arguments
        elif agent_type == 'RecurrentPPO':
            self.agent = RecurrentPPO("MlpLstmPolicy", self.environment, verbose=1)
        else:
            raise ValueError("Invalid agent type")

    def learn(self, num_steps: int = 4000):
        """
        Train the agent.
        :param num_steps: The number of steps to train the agent for.
        :return:
        """
        self.agent.learn(num_steps, callback=self.eval_callback)
    def save(self, path: str):
        """
        Save the agent to a file.
        :param path:
        :return:
        """
        self.agent.save(path)

    def predict(self, X, deterministic: bool = True):
        """
        Predict the target for the input data.
        :param X: The input data.
        :param deterministic: Whether to use deterministic or stochastic predictions.
        :return: A list of predictions, for each sequence in the input data. Use the
        get_sequencer method to get the sequencer used to sequence the data and sequence any new data.
        """

        # Sequence the data and preprocess it
        sequencer = self.environment.get_sequencer()
        sequences, targets = sequencer.sequence_by_ranges(X)
        sequences = sequencer.normalize(sequences)
        sequences = sequencer.pad_sequences(sequencer.get_max_sequence_length(), sequences)
        if self.flatten:
            sequences = [sequence.to_numpy().reshape(-1, ) for sequence in sequences]

        # Cast to float32, just in case
        sequences = [sequence.astype(np.float32) for sequence in sequences]
        for i in range(len(sequences)):
            sequences[i] = sequences[i] if isinstance(sequences[i], np.ndarray) else sequences[i].to_numpy()
        predictions = []

        # Predict the best features to select for each sequence
        for sequence in sequences:
            prediction = self.agent.predict(sequence, deterministic=deterministic)[0]
            predictions.append(prediction)
        return predictions

    def get_sequencer(self):
        """
        :return: The sequencer used by the agent.
        """
        return self.environment.get_sequencer()

    def train_models_for_ranges(self, X_train, y_train, X_test, y_test,
                                model, metrics: list[callable]):
        """
        Train a model for each sequence in the provided data, by the features that the agent
        selects for each sequence.
        :param X_train: The data to train on.
        :param y_train: The targets to train on.
        :param X_test: The data to test on.
        :param y_test: The targets to test on.
        :param model: The model to train on each sequence.
        :param metrics: A list of metrics to measure the performance of the model.
        :return: A list of models, a list of dicts for the metrics measured for each model,
        a list containing the predictions for each sequence, a list of the test sequences and a list of the test targets
        """

        # Get the predictions for the features to select
        predictions = self.predict(X_train)

        # Sequence the training and test data
        sequencer = self.environment.get_sequencer()
        sequences, targets = sequencer.sequence_by_ranges(X_train, y_train)
        test_sequences, test_targets = sequencer.sequence_by_ranges(X_test, y_test)
        models = []
        models_metrics = []
        adjusted_train_sequences = []
        for i in range(len(predictions)):
            # Create a new uninitialized model for each sequence
            model_i = model.__class__()

            # Get the sequences and targets for the current sequence
            X = sequences[i]
            y = targets[i]

            # Select the features that the agent selects
            X = X.loc[~(X == 0).all(axis=1)]
            X = X[X.columns[predictions[i] == 1]]
            adjusted_train_sequences.append(X)

            # Fit the model on the selected features and test it on the test data
            model_i.fit(X, y)
            X_test = test_sequences[i]
            y_test = test_targets[i]
            X_test = X_test[X.columns]
            y_pred = model_i.predict(X_test)
            model_metrics = {metric.__name__: metric(y_test, y_pred) for metric in metrics}

            # Save the model and its metrics
            models_metrics.append(model_metrics)
            models.append(model_i)
        return models, models_metrics, predictions, adjusted_train_sequences, targets, test_sequences, test_targets

    @staticmethod
    def load(path: str, agent_type: str = 'recurrent_ppo'):
        """
        Load an agent from a file.
        :param path: The path to the file.
        :param agent_type: The type of agent to load. Can be 'A2C', 'PPO' or 'RecurrentPPO'.
        :return: The loaded agent.
        """
        if agent_type == 'A2C':
            return A2C.load(path)
        elif agent_type == 'PPO':
            return PPO.load(path)
        elif agent_type == 'RecurrentPPO':
            return RecurrentPPO.load(path)
        else:
            raise ValueError("Invalid agent type")
