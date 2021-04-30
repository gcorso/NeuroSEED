import copy
import os
import random
import string
import unittest
from contextlib import redirect_stdout
from os import path
import io
import torch

from closest_string.task.dataset_generator_synthetic import ClosestStringDatasetGenerator
from edit_distance.models.cnn.model import CNN
from edit_distance.models.feedforward.model import MLPEncoder
from edit_distance.models.recurrent.model import GRU
from edit_distance.models.transformer.model import Transformer
from edit_distance.train import general_arg_parser, execute_train
from util.data_handling.string_generator import IndependentGenerator
from edit_distance.task.dataset_generator_synthetic import EditDistanceDatasetGenerator

ALPHABET_SIZE = 4


def generate_dataset_and_parser():
    folder_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    generator = IndependentGenerator(alphabet_size=ALPHABET_SIZE, seed=0)
    edit_dataset_name = folder_name + '/test_ed.pkl'
    edit_dataset = EditDistanceDatasetGenerator(
        N_batches={"train": 2, "val": 2, "test": 2},
        batch_size={"train": 5, "val": 3, "test": 3},
        len_sequence={"train": 10, "val": 10, "test": 10},
        max_changes={"train": 2, "val": 2, "test": 2},
        string_generator=generator, seed=0)
    edit_dataset.save_as_pickle(edit_dataset_name)

    closest_dataset_name = folder_name + '/test_closest.pkl'
    closest_dataset = ClosestStringDatasetGenerator(N_reference=3, N_query=4, len_sequence=10, min_changes=2,
                                                    max_changes=4, initials=3, string_generator=generator, seed=0)
    closest_dataset.save_as_pickle(closest_dataset_name)

    parser = general_arg_parser()
    args = parser.parse_args()
    args.data = edit_dataset_name
    args.epochs = 2
    args.print_every = 1
    args.closest_data_path = closest_dataset_name
    return folder_name, edit_dataset_name, closest_dataset_name, args


def remove_files(folder_name, edit_dataset_name, closest_dataset_name):
    if path.exists('MLPEncoder.pkl'): os.remove('MLPEncoder.pkl')
    if path.exists('0.pkl'): os.remove('0.pkl')
    if path.exists('1.pkl'): os.remove('1.pkl')
    os.remove(edit_dataset_name)
    os.remove(closest_dataset_name)
    os.rmdir(folder_name)


class TestClosestDatasetGenerationSynthetic(unittest.TestCase):

    def test_cosine_distance_stdout(self):
        folder_name, edit_dataset_name, closest_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'cosine'

        # run method storing output
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(model_class=MLPEncoder,
                          model_args=dict(layers=2,
                                          hidden_size=5,
                                          batch_norm=True),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert 'Top1:' in out and 'Top5:' in out and 'Top10:' in out, 'Wrong output format for cosine distance'

        # remove files
        remove_files(folder_name, edit_dataset_name, closest_dataset_name)

    def test_euclidean_distance_stdout(self):
        folder_name, edit_dataset_name, closest_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'euclidean'

        # run method storing output
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(model_class=MLPEncoder,
                          model_args=dict(layers=2,
                                          hidden_size=5,
                                          batch_norm=True),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert 'Top1:' in out and 'Top5:' in out and 'Top10:' in out, 'Wrong output format for euclidean distance'

        # remove files
        remove_files(folder_name, edit_dataset_name, closest_dataset_name)


    def test_square_distance_stdout(self):
        folder_name, edit_dataset_name, closest_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'square'

        # run method storing output
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(model_class=MLPEncoder,
                          model_args=dict(layers=2,
                                          hidden_size=5,
                                          batch_norm=True),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert 'Top1:' in out and 'Top5:' in out and 'Top10:' in out, 'Wrong output format for square distance'

        # remove files
        remove_files(folder_name, edit_dataset_name, closest_dataset_name)

    def test_manhattan_distance_stdout(self):
        folder_name, edit_dataset_name, closest_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'manhattan'

        # run method storing output
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(model_class=MLPEncoder,
                          model_args=dict(layers=2,
                                          hidden_size=5,
                                          batch_norm=True),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert 'Top1:' in out and 'Top5:' in out and 'Top10:' in out, 'Wrong output format for manhattan distance'

        # remove files
        remove_files(folder_name, edit_dataset_name, closest_dataset_name)

    def test_hyperbolic_distance_stdout(self):
        folder_name, edit_dataset_name, closest_dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'hyperbolic'

        # run method storing output
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(model_class=MLPEncoder,
                          model_args=dict(layers=2,
                                          hidden_size=5,
                                          batch_norm=True),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert 'Top1:' in out and 'Top5:' in out and 'Top10:' in out, 'Wrong output format for hyperbolic distance'

        # remove files
        remove_files(folder_name, edit_dataset_name, closest_dataset_name)
