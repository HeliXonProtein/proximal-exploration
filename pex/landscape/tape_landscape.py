import os
import json
import tape
import torch
import numpy as np
from . import register_landscape

@register_landscape("tape")
class TAPE_Landscape:
    """
        A TAPE-based oracle model to simulate protein fitness landscape.
    """
    
    def __init__(self, args):
        task_dir_path = os.path.join('./landscape_params/tape_landscape', args.task)
        assert os.path.exists(os.path.join(task_dir_path, 'pytorch_model.bin'))
        self.model = tape.ProteinBertForValuePrediction.from_pretrained(task_dir_path)
        with open(os.path.join(task_dir_path, 'starting_sequence.json')) as f:
            self.starting_sequence = json.load(f)
        
        self.tokenizer = tape.TAPETokenizer(vocab='iupac')
        self.device = args.device
        self.model.to(self.device)

    def get_fitness(self, sequences):
        # Input:  - sequences:      [query_batch_size, sequence_length]
        # Output: - fitness_scores: [query_batch_size]
        
        self.model.eval()
        fitness_scores = []
        for seq in sequences:
            inputs = torch.tensor(np.array([self.tokenizer.encode(seq)]))
            fitness_scores.append(self.model(inputs.to(self.device))[0].item())
        return fitness_scores
