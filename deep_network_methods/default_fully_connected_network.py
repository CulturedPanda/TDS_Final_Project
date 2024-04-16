from torch.nn import Linear, ReLU, LayerNorm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DefaultFullyConnectedNetwork(BaseFeaturesExtractor):

    def __init__(self, observation_space, input_size, features_dim):
        super(DefaultFullyConnectedNetwork, self).__init__(observation_space, features_dim)
        next_layer_size = input_size // 2
        self.layers = []
        self.layers.append(Linear(input_size, next_layer_size))
        self.layers.append(ReLU())
        steps = 0
        while next_layer_size > features_dim:
            input_size = next_layer_size
            next_layer_size = input_size // 2
            self.layers.append(Linear(input_size, next_layer_size))
            if steps % 2 == 0:
                self.layers.append(LayerNorm(next_layer_size))
            self.layers.append(ReLU())
            steps += 1
        self.layers.append(Linear(next_layer_size, features_dim))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
