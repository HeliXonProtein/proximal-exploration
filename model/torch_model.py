import torch
import numpy as np
from utils.seq_utils import sequences_to_tensor

class TorchModel:
    def __init__(self, args, alphabet, net, **kwargs):
        self.args = args
        self.alphabet = alphabet
        self.device = self.device = args.device
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        self.loss_func = torch.nn.MSELoss()

    def get_data_loader(self, sequences, labels):
        # Input:  - sequences:    [dataset_size, sequence_length]
        #         - labels:       [dataset_size]
        # Output: - loader_train: torch.utils.data.DataLoader
        
        one_hots = sequences_to_tensor(sequences, self.alphabet).float()
        labels = torch.from_numpy(labels).float()
        dataset_train = torch.utils.data.TensorDataset(one_hots, labels)
        loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True)
        return loader_train

    def compute_loss(self, data):
        # Input:  - one_hots: [batch_size, alphabet_size, sequence_length]
        #         - labels:   [batch_size]
        # Output: - loss:     [1]
        
        one_hots, labels = data
        outputs = torch.squeeze(self.net(one_hots.to(self.device)), dim=-1)
        loss = self.loss_func(outputs, labels.to(self.device))
        return loss

    def train(self, sequences, labels):
        # Input: - sequences: [dataset_size, sequence_length]
        #        - labels:    [dataset_size]
        
        self.net.train()
        loader_train = self.get_data_loader(sequences, labels)
        best_loss, num_no_improvement = np.inf, 0
        while num_no_improvement < self.args.patience:
            loss_List = []
            for data in loader_train:
                loss = self.compute_loss(data)
                loss_List.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            current_loss = np.mean(loss_List)
            if current_loss < best_loss:
                best_loss = current_loss
                num_no_improvement = 0
            else:
                num_no_improvement += 1
    
    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]
        
        self.net.eval()
        with torch.no_grad():
            one_hots = sequences_to_tensor(sequences, self.alphabet).to(self.device)
            predictions = self.net(one_hots).cpu().numpy()
        predictions = np.squeeze(predictions, axis=-1)
        return predictions
