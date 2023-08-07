import torch
import torch.nn as nn
from . import torch_model, register_model

class RNN(nn.Module):
    """
        The RNN architecture is adopted from the following paper with slight modification:
        - "Effective Surrogate Models for Protein Design with Bayesian Optimization"
          Nate Gruver, Samuel Stanton, Polina Kirichenko, Marc Finzi, Phillip Maffettone, Vivek Myers, Emily Delaney, Peyton Greenside, Andrew Gordon Wilson
          ICML Workshop on Computational Biology (2021)
          https://icml-compbio.github.io/icml-website-2021/2021/papers/WCBICML2021_paper_61.pdf
    """
    
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.h_0 = nn.Parameter(torch.zeros(num_layers*2, 1, hidden_size), requires_grad=False)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.dense = torch.nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        # Input:  [batch_size, input_size, sequence_length]
        # Output: [batch_size, 1]
        
        x, h_n = self.gru(x.permute([0, 2, 1]), self.h_0.repeat([1, x.shape[0], 1]))
        x = self.dense(x.mean(dim=1))
        return x

@register_model("rnn")
class RecurrentNetworkModel(torch_model.TorchModel):
    def __init__(self, args, alphabet, **kwargs):
        super().__init__(args, alphabet, net=RNN(input_size=len(alphabet)))
