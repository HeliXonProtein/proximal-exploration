import numpy as np

ensemble_rules = {
    'mean': lambda x: np.mean(x, axis=0),
    'lcb': lambda x: np.mean(x, axis=0) - np.std(x, axis=0),
    'ucb': lambda x: np.mean(x, axis=0) + np.std(x, axis=0)
}

class Ensemble:
    def __init__(self, models, ensemble_rule):
        self.models = models
        self.ensemble_func = ensemble_rules[ensemble_rule]
    
    def train(self, sequences, labels):
        for model in self.models:
            model.train(sequences, labels)

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]
        
        return self.ensemble_func([model.get_fitness(sequences) for model in self.models])
