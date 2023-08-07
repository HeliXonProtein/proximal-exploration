import torch
import torch.nn.functional as F
import numpy as np

def hamming_distance(seq_1, seq_2):
    return sum([x!=y for x, y in zip(seq_1, seq_2)])

def random_mutation(sequence, alphabet, num_mutations):
    wt_seq = list(sequence)
    for _ in range(num_mutations):
        idx = np.random.randint(len(sequence))
        wt_seq[idx] = alphabet[np.random.randint(len(alphabet))]
    new_seq = ''.join(wt_seq)
    return new_seq

def sequence_to_one_hot(sequence, alphabet):
    # Input:  - sequence: [sequence_length]
    #         - alphabet: [alphabet_size]
    # Output: - one_hot:  [sequence_length, alphabet_size]
    
    alphabet_dict = {x: idx for idx, x in enumerate(alphabet)}
    one_hot = F.one_hot(torch.tensor([alphabet_dict[x] for x in sequence]).long(), num_classes=len(alphabet))
    return one_hot

def sequences_to_tensor(sequences, alphabet):
    # Input:  - sequences: [batch_size, sequence_length]
    #         - alphabet:  [alphabet_size]
    # Output: - one_hots:  [batch_size, alphabet_size, sequence_length]
    
    one_hots = torch.stack([sequence_to_one_hot(seq, alphabet) for seq in sequences], dim=0)
    one_hots = torch.permute(one_hots, [0, 2, 1]).float()
    return one_hots

def sequences_to_mutation_sets(sequences, alphabet, wt_sequence, context_radius):
    # Input:  - sequences:          [batch_size, sequence_length]
    #         - alphabet:           [alphabet_size]
    #         - wt_sequence:        [sequence_length]
    #         - context_radius:     integer
    # Output: - mutation_sets:      [batch_size, max_mutation_num, alphabet_size, 2*context_radius+1]
    #         - mutation_sets_mask: [batch_size, max_mutation_num]

    context_width = 2 * context_radius + 1
    max_mutation_num = max(1, np.max([hamming_distance(seq, wt_sequence) for seq in sequences]))
    
    mutation_set_List, mutation_set_mask_List = [], []
    for seq in sequences:
        one_hot = sequence_to_one_hot(seq, alphabet).numpy()
        one_hot_padded = np.pad(one_hot, ((context_radius, context_radius), (0, 0)), mode='constant', constant_values=0.0)
        
        mutation_set = [one_hot_padded[i:i+context_width] for i in range(len(seq)) if seq[i]!=wt_sequence[i]]
        padding_len = max_mutation_num - len(mutation_set)
        mutation_set_mask = [1.0] * len(mutation_set) + [0.0] * padding_len
        mutation_set += [np.zeros(shape=(context_width, len(alphabet)))] * padding_len
            
        mutation_set_List.append(mutation_set)
        mutation_set_mask_List.append(mutation_set_mask)
    
    mutation_sets = torch.tensor(np.array(mutation_set_List)).permute([0, 1, 3, 2]).float()
    mutation_sets_mask = torch.tensor(np.array(mutation_set_mask_List)).float()
    return mutation_sets, mutation_sets_mask
