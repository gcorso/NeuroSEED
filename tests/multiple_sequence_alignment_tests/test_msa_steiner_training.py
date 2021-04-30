import copy
import io
import os
import random
import string
import unittest
from os import path
from contextlib import redirect_stdout

import torch

from multiple_alignment.steiner_string.models.convolutional.model import CNNEncoder, CNNDecoder
from multiple_alignment.steiner_string.models.mlp.model import MLPEncoder, MLPDecoder
from multiple_alignment.steiner_string.models.recurrent.model import GRUEncoder, GRUDecoder
from multiple_alignment.steiner_string.parser import general_arg_parser
from multiple_alignment.steiner_string.task.dataset_generator_genome import MSAPairDatasetGeneratorGenome
from multiple_alignment.steiner_string.train import execute_train
from tests.edit_distance_tests.test_ed_dataset_generation_genomic import generate_random_dna

ALPHABET_SIZE = 4


def generate_dataset_and_parser():
    folder_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    dataset_name = folder_name + '/test_msa_model.pkl'
    strings = [generate_random_dna(50)] + [generate_random_dna(random.randint(10, 50)) for _ in range(39)]
    sequences = {
        'train': strings[:10],
        'val': strings[10:15],
        'val_msa': [strings[15:20], strings[20:25]],
        'test': [strings[25:30], strings[30:35], strings[35:]]
    }
    dataset = MSAPairDatasetGeneratorGenome(strings=sequences, length=50)
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


class TestMSASteinerTraining(unittest.TestCase):

    def test_mlp_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=MLPEncoder, decoder_class=MLPDecoder,
                          encoder_args=dict(hidden_size=5,
                                            layers=2),
                          decoder_args=dict(hidden_size=5,
                                            layers=2),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)

    def test_cnn_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=CNNEncoder, decoder_class=CNNDecoder,
                          encoder_args=dict(readout_layers=1,
                                            channels=4,
                                            layers=2,
                                            kernel_size=3,
                                            non_linearity=True),
                          decoder_args=dict(readout_layers=1,
                                            channels=4,
                                            layers=2,
                                            kernel_size=3,
                                            non_linearity=True),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)

    def test_gru_model_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=GRUEncoder, decoder_class=GRUDecoder,
                          encoder_args=dict(recurrent_layers=1,
                                            hidden_size=5),
                          decoder_args=dict(recurrent_layers=1,
                                            hidden_size=5,
                                            reverse='True'),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)

    def test_square_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'square'
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=MLPEncoder, decoder_class=MLPDecoder,
                          encoder_args=dict(hidden_size=5,
                                            layers=2),
                          decoder_args=dict(hidden_size=5,
                                            layers=2),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)

    def test_cosine_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'cosine'
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=MLPEncoder, decoder_class=MLPDecoder,
                          encoder_args=dict(hidden_size=5,
                                            layers=2),
                          decoder_args=dict(hidden_size=5,
                                            layers=2),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)

    def test_manhattan_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'manhattan'
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=MLPEncoder, decoder_class=MLPDecoder,
                          encoder_args=dict(hidden_size=5,
                                            layers=2),
                          decoder_args=dict(hidden_size=5,
                                            layers=2),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)

    def test_hyperbolic_distance_output(self):
        folder_name, dataset_name, args = generate_dataset_and_parser()

        args = copy.copy(args)
        args.distance = 'hyperbolic'
        args.scaling = True
        f = io.StringIO()
        with redirect_stdout(f):
            execute_train(encoder_class=MLPEncoder, decoder_class=MLPDecoder,
                          encoder_args=dict(hidden_size=5,
                                            layers=2),
                          decoder_args=dict(hidden_size=5,
                                            layers=2),
                          args=args)
        out = f.getvalue()

        # check correct output
        assert "Cost multiple" in out and "Final multiple val" in out and "Final loss_val" in out \
               and "Final loss_train" in out, 'Wrong output format'

        remove_files(folder_name, dataset_name)
