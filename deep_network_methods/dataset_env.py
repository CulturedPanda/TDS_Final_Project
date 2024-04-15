import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box
import numpy as np

from .batcher import Batcher


class DatasetEnv(gym.Env):

    def __init__(self, X_train, y_train, X_prod, y_prod, downstream_model, loss_function: callable,
                 batch_size: int = 64, static_selection: bool = True):
        # The environment is defined by the training and production data, the downstream model, the loss function and
        # the batch size
        self.X_train = X_train
        self.y_train = y_train
        self.X_prod = X_prod
        self.y_prod = y_prod
        self.action_space = MultiBinary(len(X_train.columns))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(batch_size * len(X_train.columns),))
        self.downstream_model = downstream_model
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.preprocess_data()
        self.batcher = Batcher(X_train, batch_size, dtype=np.dtype('float32'))

    def preprocess_data(self):
        """
        Convert values in the data from boolean to 1s and 0s,
        and convert all values to floats
        :return:
        """
        self.X_train = self.X_train.astype(np.dtype('float32'))
        self.y_train = self.y_train.astype(np.dtype('float32'))
        self.X_prod = self.X_prod.astype(np.dtype('float32'))
        self.y_prod = self.y_prod.astype(np.dtype('float32'))

    def step(self, action):
        # Pick the features that are selected by the action
        X = self.X_train[self.X_train.columns[action == 1]]
        # Reset the downstream model's weights
        self.downstream_model = self.downstream_model.__class__()
        # Fit and test the downstream model
        self.downstream_model.fit(X, self.y_train)
        X_prod = self.X_prod[self.X_prod.columns[action == 1]]
        y_pred = self.downstream_model.predict(X_prod)
        loss = self.loss_function(self.y_prod, y_pred)
        # The reward is the negative of the loss
        reward = float(-loss)
        info = {}
        return self.batcher.__next__(), reward, self.batcher.is_done(), False, info

    def reset(self, seed=None, options=None):
        # Reset the downstream model's weights
        self.downstream_model = self.downstream_model.__class__()
        info = {}
        return self.batcher.reset(), info
