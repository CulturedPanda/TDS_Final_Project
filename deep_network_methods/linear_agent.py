import numpy as np
import os

from .linear_model_dataset_env import LinearModelDatasetEnv
from .default_fully_connected_network import DefaultFullyConnectedNetwork
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import Env
from stable_baselines3.common.callbacks import EvalCallback

from .util_funcs import is_boolean_column


class LinearAgent():
    """
    An agent that learns to select features from a dataset using a linear model.
    Unlike the sequential agent, this agent only uses a linear model to select features,
    and does not split the data into sequences.
    """

    def __init__(self, X_train, y_train, downstream_model, loss_function: callable,
                 batch_size: int = 64, agent_type: str = 'A2C',
                 network_architecture_class: BaseFeaturesExtractor | str = None,
                 environment: Env = None, network_kwargs: dict = None,
                 save_path: str = None, eval_freq: int = 500):
        """
        Initialize the agent
        :param X_train: The training data
        :param y_train: The training labels
        :param downstream_model: Downstream model to be used for training
        :param loss_function: Loss function for the downstream model
        :param batch_size: The size of the batch to use
        :param agent_type: The type of agent to use. Can be 'A2C' or 'PPO'
        :param network_architecture_class: The class of the network architecture to use.
        If None, a default fully connected network is used. If 'auto', the network architecture is determined
        automatically by the Stable Baselines3 library's defaults.
        :param environment: A gymnasium environment to use. If None, a default environment is created.
        :param network_kwargs: Arguments to pass to the network architecture class. If None, default arguments are used.
        :param save_path: The path to save the best model to.
        :param eval_freq: The frequency at which to evaluate the model.
        """
        X_train, y_train = self.preprocess_training_data(X_train, y_train)

        # If no environment is specified, create a default environment
        if environment is None:
            self.environment = LinearModelDatasetEnv(X_train, y_train, downstream_model, loss_function, batch_size)
        else:
            self.environment = environment
        self.agent_type = agent_type

        # If no network architecture is specified, use the default fully connected network
        if network_architecture_class is None:
            self.network_architecture_class = DefaultFullyConnectedNetwork
            network_kwargs = dict(
                input_size=X_train.shape[1] * batch_size,
                features_dim=X_train.shape[1],
            )
        else:
            self.network_architecture_class = network_architecture_class

        # If the network architecture is set to 'auto', use the default network architecture of the agent
        if self.network_architecture_class == "auto":
            policy_kwargs = None
        else:
            policy_kwargs = dict(
                features_extractor_class=self.network_architecture_class,
                features_extractor_kwargs=network_kwargs,
            )
        self.save_path = save_path
        self.eval_callback = EvalCallback(self.environment, best_model_save_path=save_path, log_path=save_path,
                                          eval_freq=eval_freq,
                                          deterministic=True, render=False)

        # Create the agent
        if agent_type == 'A2C':
            if policy_kwargs is None:
                self.agent = A2C("MlpPolicy", self.environment, verbose=1)
            else:
                self.agent = A2C("MlpPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)
        elif agent_type == 'PPO':
            if policy_kwargs is None:
                self.agent = PPO("MlpPolicy", self.environment, verbose=1)
            else:
                self.agent = PPO("MlpPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)


    def preprocess_training_data(self, X_train, y_train):
        """
        Convert all the data to float32, then normalize the training data
        :return: The preprocessed training data
        """
        X_train = X_train.astype(np.dtype('float32'))
        y_train = y_train.astype(np.dtype('float32'))
        # Normalize the data, while avoiding boolean columns
        for column in X_train.columns:
            if not is_boolean_column(X_train[column]):
                X_train[column] = (X_train[column] - X_train[column].mean()) / X_train[column].std()
        return X_train, y_train

    def preprocess_input_data(self, X):
        """
        Convert all the data to float32, then normalize the data
        :return: The preprocessed data
        """
        X = X.astype(np.dtype('float32'))
        # Normalize the data, while avoiding boolean columns
        for column in X.columns:
            if not is_boolean_column(X[column]):
                X[column] = (X[column] - X[column].mean()) / X[column].std()
        X = X.to_numpy()
        return X.reshape(-1, )


    def learn(self, num_steps: int = 4000):
        """
        Train the agent.
        :param num_steps: The number of steps / episodes to train the agent for.
        :return:
        """
        self.agent.learn(num_steps, callback=self.eval_callback)

    def predict(self, X, deterministic: bool = True):
        """
        Predict the best features to select from the data.
        :param X: The data to predict on.
        :param deterministic: Whether to use a deterministic policy or not.
        :return: The best features to select.
        """
        X = self.preprocess_input_data(X)
        return self.agent.predict(X, deterministic=deterministic)[0]

    def save(self, model_name: str = None):
        """
        Save the model to a file.
        :param model_name: The name of the model to save.
        :return:
        """
        save_path = os.path.join(self.save_path, model_name)
        self.agent.save(save_path)

    @staticmethod
    def load(save_path: str, agent_type='A2C'):
        """
        Load a model from a file.
        :param save_path: The path to the saved model.
        :param agent_type: The type of agent to load. It Can be 'A2C' or 'PPO'
        :return: The loaded model.
        """
        if agent_type == 'A2C':
            return A2C.load(save_path)
        elif agent_type == 'PPO':
            return PPO.load(save_path)
        else:
            raise ValueError("Invalid agent type")

