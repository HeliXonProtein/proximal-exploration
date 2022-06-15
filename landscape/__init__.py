import os
import importlib

protein_alphabet = 'ACDEFGHIKLMNPQRSTVWY'

task_collection = {
    'avGFP': protein_alphabet,
    'AAV': protein_alphabet,
    'TEM': protein_alphabet,
    'E4B': protein_alphabet,
    'AMIE': protein_alphabet,
    'LGK': protein_alphabet,
    'Pab1': protein_alphabet,
    'UBE2I': protein_alphabet
}

landscape_collection = {}

def register_landscape(landscape_name):
    def register_func(landscape_class):
        landscape_collection[landscape_name] = landscape_class
        return landscape_class
    return register_func

def get_landscape(args):
    landscape = landscape_collection[args.oracle_model](args)
    return landscape, task_collection[args.task], landscape.starting_sequence

for file_name in os.listdir(os.path.dirname(__file__)):
    if file_name.endswith('.py') and not file_name.startswith('_'):
        importlib.import_module('landscape.' + file_name[:-3])
