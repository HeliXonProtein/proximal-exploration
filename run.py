import numpy as np
from landscape import get_landscape, task_collection, landscape_collection
from algorithm import get_algorithm, algorithm_collection
from model import get_model, model_collection
from model.ensemble import ensemble_rules
from utils.os_utils import get_arg_parser
from utils.eval_utils import Runner

def get_args():
    parser = get_arg_parser()
    
    parser.add_argument('--device', help='device', type=str, default='cuda')
    
    # landscape arguments
    parser.add_argument('--task', help='fitness landscape', type=str, default='avGFP', choices=task_collection.keys())
    parser.add_argument('--oracle_model', help='oracle model of fitness landscape', type=str, default='tape', choices=landscape_collection.keys())

    # algorithm arguments
    parser.add_argument('--alg', help='exploration algorithm', type=str, default='pex', choices=algorithm_collection.keys())
    parser.add_argument('--num_rounds', help='number of query rounds', type=np.int32, default=10)
    parser.add_argument('--num_queries_per_round', help='number of black-box queries per round', type=np.int32, default=100)
    parser.add_argument('--num_model_queries_per_round', help='number of model predictions per round', type=np.int32, default=2000)
    
    # model arguments
    parser.add_argument('--net', help='surrogate model architecture', type=str, default='mufacnet', choices=model_collection.keys())
    parser.add_argument('--lr', help='learning rate', type=np.float32, default=1e-3)
    parser.add_argument('--batch_size', help='batch size', type=np.int32, default=256)
    parser.add_argument('--patience', help='number of epochs without improvement to wait before terminating training', type=np.int32, default=10)
    parser.add_argument('--ensemble_size', help='number of model instances in ensemble', type=np.int32, default=3)
    parser.add_argument('--ensemble_rule', help='rule to aggregate the ensemble predictions', type=str, default='mean', choices=ensemble_rules.keys())

    args, _ = parser.parse_known_args()
    
    # PEX arguments
    if args.alg == 'pex':
        parser.add_argument('--num_random_mutations', help='number of amino acids to mutate per sequence', type=np.int32, default=2)
        parser.add_argument('--frontier_neighbor_size', help='size of the frontier neighbor', type=np.int32, default=5)
    
    # MuFacNet arguments
    if args.net == 'mufacnet':
        parser.add_argument('--latent_dim', help='dimension of latent mutation embedding', type=np.int32, default=32)
        parser.add_argument('--context_radius', help='the radius of context window', type=np.int32, default=10)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    
    landscape, alphabet, starting_sequence = get_landscape(args)
    model = get_model(args, alphabet=alphabet, starting_sequence=starting_sequence)
    explorer = get_algorithm(args, model=model, alphabet=alphabet, starting_sequence=starting_sequence)

    runner = Runner(args)
    runner.run(landscape, starting_sequence, model, explorer)
