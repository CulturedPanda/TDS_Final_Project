import pandas as pd
import tensorflow as tf
from keras import __version__
tf.keras.__version__ = __version__
from keras.layers import Dense
from keras.models import Sequential
from .dataset_env import DatasetEnv
from .agent_base import AgentBase

class FullyConnectedSelector(Sequential):

    def __init__(self, dataset: pd.DataFrame, targets: pd.Series,
                 downstream_model, loss_function, batch_size: int = 64, static_selection: bool = True):
        super(FullyConnectedSelector, self).__init__()
        self.dataset = dataset
        self.targets = targets
        self.batch_size = batch_size

        input_size = output_size = len(dataset.columns)
        self.add(Dense(input_size, input_shape=(input_size,), activation='relu'))
        layer_size = input_size // 2
        while layer_size > output_size:
            self.add(Dense(layer_size, activation='relu'))
            layer_size = layer_size // 2
        self.add(Dense(output_size, activation='sigmoid'))
        self.environment = DatasetEnv(dataset, targets, downstream_model, loss_function, batch_size, static_selection)
        self.agent = AgentBase(self, self.environment)

    def __call__(self, *args, **kwargs):
        output = super(FullyConnectedSelector, self).__call__(*args, **kwargs)
        return tf.reshape(output, (self.batch_size, len(self.dataset.columns)))

    def train(self, nb_steps):
        self.agent.train(nb_steps)
