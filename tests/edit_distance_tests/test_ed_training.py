import copy
import os
import random
import string
import unittest
from os import path

import torch

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
    dataset_name = folder_name + '/test_ed_model.pkl'
    dataset = EditDistanceDatasetGenerator(
        N_batches={"train": 2, "val": 2, "test": 2},
        batch_size={"train": 5, "val": 3, "test": 3},
        len_sequence={"train": 10, "val": 10, "test": 10},
        max_changes={"train": 2, "val": 2, "test": 2},
        string_generator=generator, seed=0)
    dataset.save_as_pickle(dataset_name)

    parser = general_arg_parser()
    args = parser.parse_args()
    args.data = dataset_name
    args.epochs = 2
    args.print_every = 1
    args.distance = "euclidean"
    return folder_name, dataset_name, args


def remove_files(folder_name, dataset_name):
    if path.exists('0.pkl'): os.remove('0.pkl')
    if path.exists('1.pkl'): os.remove('1.pkl')
    os.remove(dataset_name)
    os.rmdir(folder_name)


class TestEDTraining(unittest.TestCase):

    def test_mlp_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        assert path.exists('MLPEncoder.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('MLPEncoder.pkl')
        remove_files(folder_name, dataset_name)

    def test_cnn_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()
        execute_train(model_class=CNN,
                      model_args=dict(readout_layers=1,
                                      channels=4,
                                      layers=2,
                                      kernel_size=3,
                                      pooling='avg',
                                      non_linearity=True,
                                      batch_norm=True,
                                      stride=1),
                      args=args)

        assert path.exists('CNN.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('CNN.pkl')
        remove_files(folder_name, dataset_name)

    def test_gru_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()
        execute_train(model_class=GRU,
                      model_args=dict(recurrent_layers=1,
                                      readout_layers=1,
                                      hidden_size=5),
                      args=args)

        assert path.exists('GRU.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('GRU.pkl')
        remove_files(folder_name, dataset_name)

    def test_transformer_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        execute_train(model_class=Transformer,
                      model_args=dict(segment_size=2,
                                      trans_layers=1,
                                      readout_layers=1,
                                      hidden_size=4,
                                      mask='empty',
                                      heads=2,
                                      layer_norm=True),
                      args=args)

        assert path.exists('Transformer.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('Transformer.pkl')
        remove_files(folder_name, dataset_name)

    def test_square_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'square'
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        assert path.exists('MLPEncoder.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('MLPEncoder.pkl')
        remove_files(folder_name, dataset_name)

    def test_cosine_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'cosine'
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        assert path.exists('MLPEncoder.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('MLPEncoder.pkl')
        remove_files(folder_name, dataset_name)

    def test_manhattan_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'manhattan'
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        assert path.exists('MLPEncoder.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('MLPEncoder.pkl')
        remove_files(folder_name, dataset_name)

    def test_hyperbolic_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'hyperbolic'
        args.scaling = True
        execute_train(model_class=MLPEncoder,
                      model_args=dict(layers=2,
                                      hidden_size=5,
                                      batch_norm=True),
                      args=args)

        assert path.exists('MLPEncoder.pkl')
        assert path.exists('0.pkl') or path.exists('1.pkl')

        os.remove('MLPEncoder.pkl')
        remove_files(folder_name, dataset_name)

