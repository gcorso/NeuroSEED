import copy
import os
import random
import string
import unittest
from contextlib import redirect_stdout
from os import path
import io

from edit_distance.models.feedforward.model import MLPEncoder
from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator
from edit_distance.train import general_arg_parser, execute_train
from hierarchical_clustering.task.dataset_generator_synthetic import HierarchicalClusteringDatasetGenerator
from tests.edit_distance_tests.test_ed_dataset_generation_genomic import generate_random_dna
from util.data_handling.string_generator import IndependentGenerator
from edit_distance.task.dataset_generator_synthetic import EditDistanceDatasetGenerator

ALPHABET_SIZE = 4


def generate_dataset_and_parser():
    folder_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    generator = IndependentGenerator(alphabet_size=ALPHABET_SIZE, seed=0)
    edit_dataset_name = folder_name + '/test_ed.pkl'
    strings = [generate_random_dna(50)] + [generate_random_dna(random.randint(10, 50)) for _ in range(19)]
    strings_dict = {'train': strings[:10], 'val': strings[10:15], 'test': strings[15:]}
    edit_dataset = EditDistanceGenomicDatasetGenerator(strings=strings_dict)
    edit_dataset.save_as_pickle(edit_dataset_name)

    parser = general_arg_parser()
    args = parser.parse_args()
    args.data = edit_dataset_name
    args.epochs = 2
    args.print_every = 1
    args.construct_msa_tree = 'True'
    return folder_name, edit_dataset_name, args


def remove_files(folder_name, edit_dataset_name):
    if path.exists('MLPEncoder.pkl'): os.remove('MLPEncoder.pkl')
    if path.exists('0.pkl'): os.remove('0.pkl')
    if path.exists('1.pkl'): os.remove('1.pkl')
    if path.exists('njtree.dnd'): os.remove('njtree.dnd')
    if path.exists('sequences.fasta'): os.remove('sequences.fasta')
    os.remove(edit_dataset_name)
    os.rmdir(folder_name)


class TestMSAGuideTree(unittest.TestCase):

    def test_tree_output(self):
        folder_name, edit_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'cosine'

        # run method storing output
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        # check output tree
        assert path.exists('njtree.dnd'), "Tree file missing"

        # remove files
        remove_files(folder_name, edit_dataset_name)

    def test_sequence_output(self):
        folder_name, edit_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'cosine'

        # run method storing output
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        # check output tree
        assert path.exists('sequences.fasta'), "Sequences file missing"

        # remove files
        remove_files(folder_name, edit_dataset_name)