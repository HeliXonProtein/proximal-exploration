import random
import numpy as np
from . import register_algorithm
from utils.seq_utils import hamming_distance, random_mutation

@register_algorithm("pex")
class ProximalExploration:
    """
        Proximal Exploration (PEX)
    """
    
    def __init__(self, args, model, alphabet, starting_sequence):
        self.model = model
        self.alphabet = alphabet
        self.wt_sequence = starting_sequence
        self.num_queries_per_round = args.num_queries_per_round
        self.num_model_queries_per_round = args.num_model_queries_per_round
        self.batch_size = args.batch_size
        self.num_random_mutations = args.num_random_mutations
        self.frontier_neighbor_size = args.frontier_neighbor_size
    
    def propose_sequences(self, measured_sequences):
        # Input:  - measured_sequences: pandas.DataFrame
        #           - 'sequence':       [sequence_length]
        #           - 'true_score':     float
        # Output: - query_batch:        [num_queries, sequence_length]
        #         - model_scores:       [num_queries]
        
        query_batch = self._propose_sequences(measured_sequences)
        model_scores = np.concatenate([
            self.model.get_fitness(query_batch[i:i+self.batch_size])
            for i in range(0, len(query_batch), self.batch_size)
        ])
        return query_batch, model_scores

    def _propose_sequences(self, measured_sequences):
        measured_sequence_set = set(measured_sequences['sequence'])
        
        # Generate random mutations in the first round.
        if len(measured_sequence_set)==1:
            query_batch = []
            while len(query_batch) < self.num_queries_per_round:
                random_mutant = random_mutation(self.wt_sequence, self.alphabet, self.num_random_mutations)
                if random_mutant not in measured_sequence_set:
                    query_batch.append(random_mutant)
                    measured_sequence_set.add(random_mutant)
            return query_batch
        
        # Arrange measured sequences by the distance to the wild type.
        measured_sequence_dict = {}
        for _, data in measured_sequences.iterrows():
            distance_to_wt = hamming_distance(data['sequence'], self.wt_sequence)
            if distance_to_wt not in measured_sequence_dict.keys():
                measured_sequence_dict[distance_to_wt] = []
            measured_sequence_dict[distance_to_wt].append(data)
        
        # Highlight measured sequences near the proximal frontier.
        frontier_neighbors, frontier_height = [], -np.inf
        for distance_to_wt in sorted(measured_sequence_dict.keys()):
            data_list = measured_sequence_dict[distance_to_wt]
            data_list.sort(reverse=True, key=lambda x:x['true_score'])
            for data in data_list[:self.frontier_neighbor_size]:
                if data['true_score'] > frontier_height:
                    frontier_neighbors.append(data)
            frontier_height = max(frontier_height, data_list[0]['true_score'])

        # Construct the candiate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        candidate_pool = []
        while len(candidate_pool) < self.num_model_queries_per_round:
            candidate_sequence = random_mutation(random.choice(frontier_neighbors)['sequence'], self.alphabet, self.num_random_mutations)
            if candidate_sequence not in measured_sequence_set:
                candidate_pool.append(candidate_sequence)
                measured_sequence_set.add(candidate_sequence)
        
        # Arrange the candidate pool by the distance to the wild type.
        candidate_pool_dict = {}
        for i in range(0, len(candidate_pool), self.batch_size):
            candidate_batch =  candidate_pool[i:i+self.batch_size]
            model_scores = self.model.get_fitness(candidate_batch)
            for candidate, model_score in zip(candidate_batch, model_scores):
                distance_to_wt = hamming_distance(candidate, self.wt_sequence)
                if distance_to_wt not in candidate_pool_dict.keys():
                    candidate_pool_dict[distance_to_wt] = []
                candidate_pool_dict[distance_to_wt].append(dict(sequence=candidate, model_score=model_score))
        for distance_to_wt in sorted(candidate_pool_dict.keys()):
            candidate_pool_dict[distance_to_wt].sort(reverse=True, key=lambda x:x['model_score'])
        
        # Construct the query batch by iteratively extracting the proximal frontier. 
        query_batch = []
        while len(query_batch) < self.num_queries_per_round:
            # Compute the proximal frontier by Andrew's monotone chain convex hull algorithm. (line 5 of Algorithm 2 in the paper)
            # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
            stack = []
            for distance_to_wt in sorted(candidate_pool_dict.keys()):
                if len(candidate_pool_dict[distance_to_wt])>0:
                    data = candidate_pool_dict[distance_to_wt][0]
                    new_point = np.array([distance_to_wt, data['model_score']])
                    def check_convex_hull(point_1, point_2, point_3):
                        return np.cross(point_2-point_1, point_3-point_1) <= 0
                    while len(stack)>1 and not check_convex_hull(stack[-2], stack[-1], new_point):
                        stack.pop(-1)
                    stack.append(new_point)
            while len(stack)>=2 and stack[-1][1] < stack[-2][1]:
                stack.pop(-1)
            
            # Update query batch and candidate pool. (line 6 of Algorithm 2 in the paper)
            for distance_to_wt, model_score in stack:
                if len(query_batch) < self.num_queries_per_round:
                    query_batch.append(candidate_pool_dict[distance_to_wt][0]['sequence'])
                    candidate_pool_dict[distance_to_wt].pop(0)

        return query_batch
