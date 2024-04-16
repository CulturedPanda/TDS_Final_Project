import numpy as np
import os

from .dataset_env import DatasetEnv
from .default_fully_connected_network import DefaultFullyConnectedNetwork
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import Env
from stable_baselines3.common.callbacks import EvalCallback


class LinearAgent():
    """
    An agent that learns to select features from a dataset using a linear model.
    """

    def __init__(self, X_train, y_train, downstream_model, loss_function: callable,
                 batch_size: int = 64, agent_type: str = 'A2C',
                 network_architecture_class: BaseFeaturesExtractor | str = None,
                 environment: Env = None, network_kwargs: dict = None,
                 save_path: str = None, eval_freq: int = 500):
        """
        Initialize the agent
        :param X_train:
        :param y_train:
        :param downstream_model: Downstream model to be used for training
        :param loss_function: Loss function for the downstream model
        :param batch_size:
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
        if environment is None:
            self.environment = DatasetEnv(X_train, y_train, downstream_model, loss_function, batch_size)
        else:
            self.environment = environment
        self.agent_type = agent_type
        if network_architecture_class is None:
            self.network_architecture_class = DefaultFullyConnectedNetwork
            network_kwargs = dict(
                input_size=X_train.shape[1] * batch_size,
                features_dim=X_train.shape[1],
            )
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
                                          eval_freq=eval_freq,
                                          deterministic=True, render=False)
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
        :return:
        """
        X_train = X_train.astype(np.dtype('float32'))
        y_train = y_train.astype(np.dtype('float32'))
        X_train = (X_train - X_train.mean()) / X_train.std()
        return X_train, y_train

    def preprocess_input_data(self, X):
        """
        Convert all the data to float32, then normalize the data
        :return:
        """
        X = X.astype(np.dtype('float32'))
        X = (X - X.mean()) / X.std()
        return X


    def learn(self, num_steps: int = 4000):
        self.agent.learn(num_steps, callback=self.eval_callback)

    def predict(self, X, deterministic: bool = True):
        X = self.preprocess_input_data(X)
        return self.agent.predict(X, deterministic=deterministic)

    def save(self, model_name: str = None):
        save_path = os.path.join(self.save_path, model_name)
        self.agent.save(save_path)

    @staticmethod
    def load(save_path: str, agent_type='A2C'):
        if agent_type == 'A2C':
            return A2C.load(save_path)
        elif agent_type == 'PPO':
            return PPO.load(save_path)
        else:
            raise ValueError("Invalid agent type")

