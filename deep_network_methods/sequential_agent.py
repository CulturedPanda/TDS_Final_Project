import numpy as np
import os

from .sequential_model_dataset_env import SequentialModelDatasetEnv
from .default_fully_connected_network import DefaultFullyConnectedNetwork
from .default_sequence_modeling_network import DefaultSequenceModelingNetwork
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback


class SequentialAgent:

    def __init__(self, X_train, y_train, col: str, downstream_model, loss_function: callable,
                 agent_type: str = 'recurrent_ppo', network_type='recurrent',
                 network_architecture_class: BaseFeaturesExtractor | str = None,
                 environment: SequentialModelDatasetEnv = None, network_kwargs: dict = None,
                 save_path: str = None, eval_freq: int = 500):
        """
        Initialize the agent
        :param X_train: The training data
        :param y_train: The training targets
        :param col: The column to sequence by
        :param downstream_model: The downstream model that will be used for prediction on the data, after feature selection
        :param loss_function: The loss function to use for the downstream model
        :param agent_type: The type of agent to use. Can be 'A2C' or 'PPO' for fully connected networks,
         or 'recurrent_ppo' for recurrent networks
        :param network_type: The type of network to use. Can be 'fully_connected' or 'recurrent'
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
        if environment is None:
            self.environment = SequentialModelDatasetEnv(X_train=X_train,
                                                         y_train=y_train,
                                                         col=col,
                                                         downstream_model=downstream_model,
                                                         loss_function=loss_function)
        else:
            self.environment = environment
        self.agent_type = agent_type
        if network_architecture_class is None:
            if network_type == 'fully_connected':
                self.network_architecture_class = DefaultFullyConnectedNetwork
                network_kwargs = dict(
                    input_size=X_train.shape[1] * self.environment.observation_space.shape[0],
                    features_dim=X_train.shape[1],
                )
            elif network_type == 'recurrent':
                self.network_architecture_class = DefaultSequenceModelingNetwork
                network_kwargs = dict(
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
                                          eval_freq=eval_freq, deterministic=True)
        if agent_type == 'A2C':
            self.agent = A2C("MlpPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)
        elif agent_type == 'PPO':
            self.agent = PPO("MlpPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)
        elif agent_type == 'recurrent_ppo':
            # self.agent = RecurrentPPO("MlpLstmPolicy", self.environment, policy_kwargs=policy_kwargs, verbose=1)
            self.agent = RecurrentPPO("MlpLstmPolicy", self.environment, verbose=1)
        else:
            raise ValueError("Invalid agent type")

    def learn(self, num_steps: int = 4000):
        self.agent.learn(num_steps, callback=self.eval_callback)

    def save(self, path: str):
        self.agent.save(path)

    def predict(self, X, deterministic: bool = True, y=None, return_sequences=False):
        sequencer = self.environment.get_sequencer()
        sequences, targets = sequencer.sequence_by_ranges(X, y)
        sequences = sequencer.normalize(sequences)
        sequences = sequencer.pad_sequences(sequencer.get_max_sequence_length(), sequences)
        predictions = []
        for sequence in sequences:
            prediction = self.agent.predict(sequence, deterministic=deterministic)[0]
            predictions.append(prediction)
        if return_sequences:
            return predictions, sequences, targets
        return predictions

    def get_sequencer(self):
        return self.environment.get_sequencer()

    def train_models_for_ranges(self, X, y, model, loss_function: callable):
        """
        Train a model for each sequence in the provided data
        :param X:
        :param y:
        :param model:
        :param loss_function:
        :return:
        """
        predictions, sequences, targets = self.predict(X, y=y, return_sequences=True)
        models = []
        loss_values = []
        for i in range(len(predictions)):
            model_i = model.__class__()
            X = sequences[i]
            y = targets[i]
            X = X.loc[~(X == 0).all(axis=1)]
            X = X[X.columns[predictions[i] == 1]]
            model_i.fit(X, y)
            y_pred = model_i.predict(X)
            loss = loss_function(y, y_pred)
            loss_values.append(loss)
            models.append(model_i)
        return models

    @staticmethod
    def load(path: str, agent_type: str = 'recurrent_ppo'):
        if agent_type == 'A2C':
            return A2C.load(path)
        elif agent_type == 'PPO':
            return PPO.load(path)
        elif agent_type == 'recurrent_ppo':
            return RecurrentPPO.load(path)
        else:
            raise ValueError("Invalid agent type")
