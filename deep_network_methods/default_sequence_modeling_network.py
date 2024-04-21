from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import LSTM, Linear, Softmax


class DefaultSequenceModelingNetwork(BaseFeaturesExtractor):
    """
    A default sequence modeling network for the reinforcement learning algorithms.
    Acts as a feature extractor for the algorithms.
    """

    def __init__(self, observation_space, features_dim, num_layers=3, hidden_size=128):
        """
        :param observation_space: The observation space of the environment
        :param features_dim: The size of the output layer
        :param num_layers: The number of LSTM layers
        :param hidden_size: The size of the hidden layer in the LSTM
        """
        super(DefaultSequenceModelingNetwork, self).__init__(observation_space, features_dim)

        self.lstm = LSTM(
            input_size=features_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Downsample the output of the LSTM to the desired features dimension
        self.downsample_layer = Linear(hidden_size * 2, features_dim)

        # Attention mechanism to weight the output of the LSTM
        self.attention_weights = Linear(features_dim, 1)

    def forward(self, model_input):
        # Unbatched input is input from the environment, which is a single observation
        if model_input.shape[0] == 1:
            model_input = model_input.view(-1, self.features_dim)
            # Remove 0 padding from input by removing rows with all zeros
            mask = model_input.sum(dim=-1) != 0
            model_input = model_input[mask]
            output = self.lstm(model_input)[0]
            output = self.downsample_layer(output)

            # Compute attention weights and apply them to the output
            attention_weights = self.attention_weights(output)
            attention_weights = Softmax(dim=0)(attention_weights)
            output = (output * attention_weights).sum(dim=0, keepdims=True)
            return output
        # Batched input is input from the replay buffer, which we do not want to
        # unbatch or the algorithm will not work
        else:
            model_input = model_input.view(model_input.shape[0], -1, self.features_dim)
            output = self.lstm(model_input)[0]
            output = self.downsample_layer(output)
            attention_weights = self.attention_weights(output)
            attention_weights = Softmax(dim=1)(attention_weights)
            output = (output * attention_weights).sum(dim=1, keepdims=False)
            return output

