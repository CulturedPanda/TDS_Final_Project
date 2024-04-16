from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import LSTM, Linear
class DefaultSequenceModelingNetwork(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim, num_layers=3):
        super(DefaultSequenceModelingNetwork, self).__init__(observation_space, features_dim)

        self.lstm = LSTM(
            input_size=features_dim,
            hidden_size=features_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention_weights = Linear(features_dim, 1)

    def forward(self, model_input):
        model_input = model_input.view(-1, self.features_dim)
        # Remove 0 padding from input by removing rows with all zeros
        mask = model_input.sum(dim=-1) != 0
        model_input = model_input[mask]
        output = self.lstm(model_input)[0]
        attention_weights = self.attention_weights(output)
        attention_weights = attention_weights / attention_weights.sum(dim=0, keepdim=True)
        output = (output * attention_weights).sum(dim=0)
        return output