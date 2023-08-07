import os
import json
import torch
import torch.nn as nn
import numpy as np
from sequence_models.structure import Attention1d
from . import register_landscape

class Decoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention1d = Attention1d(in_dim=hidden_dim)
        self.dense_3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_4 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.attention1d(x)
        x = torch.relu(self.dense_3(x))
        x = self.dense_4(x)
        return x

class ESM1b_Attention1d(nn.Module):
    def __init__(self):
        super().__init__()
        esm_dir_path = './landscape_params/esm1b_landscape/esm_params'
        torch.hub.set_dir(esm_dir_path)
        self.encoder, self.alphabet = torch.hub.load('facebookresearch/esm:main', 'esm1b_t33_650M_UR50S')
        self.tokenizer = self.alphabet.get_batch_converter()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x, repr_layers=[33], return_contacts=False)["representations"][33]
        x = self.decoder(x)
        return x

@register_landscape("esm1b")
class ESM1b_Landscape:
    """
        An ESM-based oracle model to simulate protein fitness landscape.
    """
    
    def __init__(self, args):
        task_dir_path = os.path.join('./landscape_params/esm1b_landscape', args.task)
        assert os.path.exists(os.path.join(task_dir_path, 'decoder.pt'))
        self.model = ESM1b_Attention1d()
        self.model.decoder.load_state_dict(torch.load(os.path.join(task_dir_path, 'decoder.pt')))
        with open(os.path.join(task_dir_path, 'starting_sequence.json')) as f:
            self.starting_sequence = json.load(f)
        
        self.tokenizer = self.model.tokenizer
        self.device = args.device
        self.model.to(self.device)
        
    def get_fitness(self, sequences):
        # Input:  - sequences:      [query_batch_size, sequence_length]
        # Output: - fitness_scores: [query_batch_size]
        
        self.model.eval()
        fitness_scores = []
        for seq in sequences:
            inputs = self.tokenizer([('seq', seq)])[-1]
            fitness_scores.append(self.model(inputs.to(self.device)).item())
        return fitness_scores
