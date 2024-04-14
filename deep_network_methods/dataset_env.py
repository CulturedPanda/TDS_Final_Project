import copy

import pandas as pd
from gym import Env
from gym.spaces import MultiBinary, Box
import numpy as np


class DatasetEnv(Env):

    def __init__(self, dataset: pd.DataFrame, targets: pd.Series, downstream_model, loss_function: callable,
                 batch_size: int = 64, static_selection: bool = True):
        self.dataset = dataset
        self.targets = targets
        self.action_space = MultiBinary(len(dataset.columns) * batch_size)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(dataset.columns),))
        self.model_uninitialized_copy = copy.deepcopy(downstream_model)
        self.downstream_model = downstream_model
        self.loss_function = loss_function
        self.batch_size = batch_size

    def step(self, action):
        feature_selection = action.reshape(self.batch_size, len(self.dataset.columns))
        # Perform majority voting to select the features
        feature_selection = np.sum(feature_selection, axis=0) >= self.batch_size / 2
        feature_selection = feature_selection.astype(bool)
        X = self.dataset.loc[:, feature_selection]
        y = self.targets
        self.downstream_model.fit(X, y)
        y_pred = self.downstream_model.predict(X)
        loss = self.loss_function(y, y_pred)
        reward = np.invert(loss)
        done = True
        info = {}
        return X.columns, reward, done, info

    def reset(self):
        # Reset the downstream model's weights
        self.downstream_model = copy.deepcopy(self.model_uninitialized_copy)
        return self.dataset.columns
