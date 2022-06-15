import torch
import torch.nn as nn
from . import torch_model, register_model

class CNN(nn.Module):
    """
        The CNN architecture is adopted from the following paper with slight modification:
        - "AdaLead: A simple and robust adaptive greedy search algorithm for sequence design"
          Sam Sinai, Richard Wang, Alexander Whatley, Stewart Slocum, Elina Locane, Eric D. Kelsic
          arXiv preprint 2010.02141 (2020)
          https://arxiv.org/abs/2010.02141
    """
    
    def __init__(self, num_input_channels, num_filters=32, hidden_dim=128, kernel_size=5):
        super().__init__()
        self.conv_1 = nn.Conv1d(num_input_channels, num_filters, kernel_size, padding='valid')
        self.conv_2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding='same')
        self.conv_3 = nn.Conv1d(num_filters, num_filters, kernel_size, padding='same')
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dense_1 = nn.Linear(num_filters, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_1 = nn.Dropout(0.25)
        self.dense_3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Input:  [batch_size, num_input_channels, sequence_length]
        # Output: [batch_size, 1]
        
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.squeeze(self.global_max_pool(x), dim=-1)
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.dropout_1(x)
        x = self.dense_3(x)
        return x

@register_model("cnn")
class ConvolutionalNetworkModel(torch_model.TorchModel):
    def __init__(self, args, alphabet, **kwargs):
        super().__init__(args, alphabet, net=CNN(num_input_channels=len(alphabet)))
