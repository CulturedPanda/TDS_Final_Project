from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import LSTM, Linear, Softmax


class DefaultSequenceModelingNetwork(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim, num_layers=3, hidden_size=128):
        super(DefaultSequenceModelingNetwork, self).__init__(observation_space, features_dim)

        self.lstm = LSTM(
            input_size=features_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.downsample_layer = Linear(hidden_size * 2, features_dim)
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

