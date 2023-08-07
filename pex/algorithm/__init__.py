import os
import importlib

algorithm_collection = {}

def register_algorithm(algorithm_name):
    def register_func(algorithm_class):
        algorithm_collection[algorithm_name] = algorithm_class
        return algorithm_class
    return register_func

def get_algorithm(args, model, alphabet, starting_sequence):
    return algorithm_collection[args.alg](args, model, alphabet, starting_sequence)

for file_name in os.listdir(os.path.dirname(__file__)):
    if file_name.endswith('.py') and not file_name.startswith('_'):
        importlib.import_module('pex.algorithm.' + file_name[:-3])
