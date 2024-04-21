import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box
import numpy as np

from .sequencer import Sequencer


class SequentialModelDatasetEnv(gym.Env):
    """
    An environment for the sequential agent that selects features from a dataset.
    """

    def __init__(self, X_train, y_train, col, downstream_model, loss_function: callable, flatten=False):
        """
        :param X_train: The training data.
        :param y_train: The training labels.
        :param col: The column to sequence the data by.
        :param downstream_model: Downstream model to be used for training.
        :param loss_function: Loss function for the downstream model.
        :param flatten: Whether to flatten the data or not.
        """
        # Initialize the sequencer
        self.sequencer = Sequencer(X_train, y_train, col, flatten=flatten)
        self.sequencer.normalize()
        self.action_space = MultiBinary(len(X_train.columns))

        # Pad the sequences to the same length
        max_sequence_len = self.sequencer.get_max_sequence_length()
        self.sequencer.pad_sequences(max_sequence_len)

        # Define the observation space, such that it is the same length as the sequences
        if flatten:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(max_sequence_len * len(X_train.columns),))
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(max_sequence_len, len(X_train.columns)))
        self.downstream_model = downstream_model
        self.loss_function = loss_function

    def get_sequencer(self):
        """

        :return: The sequencer used by the environment
        """
        return self.sequencer

    def step(self, action):
        # Pick the features that are selected by the action
        X, y = self.sequencer.get_current()
        # Remove rows where all values are zero, as those are the padding rows
        X = X.loc[~(X == 0).all(axis=1)]
        X = X[X.columns[action == 1]]
        # Reset the downstream model's weights
        self.downstream_model = self.downstream_model.__class__()
        # Fit and test the downstream model
        self.downstream_model.fit(X, y)
        y_pred = self.downstream_model.predict(X)
        loss = self.loss_function(y, y_pred)
        # The reward is the negative of the loss
        reward = float(-loss)
        info = {}
        return self.sequencer.__next__(), reward, self.sequencer.is_done(), False, info

    def reset(self, seed=None, options=None):
        # Reset the downstream model's weights
        self.downstream_model = self.downstream_model.__class__()
        info = {}
        return self.sequencer.reset(), info
