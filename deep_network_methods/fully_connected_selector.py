# import pandas as pd
# from keras.layers import Dense, Flatten
# from keras.models import Sequential
# from .dataset_env import DatasetEnv
# # from .agent_base import AgentBase
#
# class FullyConnectedSelector(Sequential):
#
#     def __init__(self, dataset: pd.DataFrame, targets: pd.Series,
#                  downstream_model, loss_function, batch_size: int = 64, static_selection: bool = True):
#         super(FullyConnectedSelector, self).__init__()
#         self.dataset = dataset
#         self.targets = targets
#         self.batch_size = batch_size
#
#         input_size = len(dataset.columns) * batch_size
#         output_size = len(dataset.columns)
#         self.add(Flatten(input_shape=(batch_size, input_size,)))
#         self.add(Dense(input_size, activation='relu'))
#         layer_size = input_size // 2
#         while layer_size >= output_size:
#             self.add(Dense(layer_size, activation='relu'))
#             layer_size = layer_size // 2
#         self.add(Dense(output_size, activation='linear'))
#         self.environment = DatasetEnv(dataset, targets, downstream_model, loss_function, batch_size, static_selection)
#         # self.agent = AgentBase(self, self.environment)
#
#     # def train(self, nb_steps):
#     #     self.agent.train(nb_steps)
