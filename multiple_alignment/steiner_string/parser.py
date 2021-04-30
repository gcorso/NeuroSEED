import argparse


def general_arg_parser():
    """ Parsing of parameters common to all the different models """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Data')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training (GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--print_every', type=int, default=1, help='Print training results every')
    parser.add_argument('--eval_every', type=int, default=1, help='Print training results every')
    parser.add_argument('--expid', type=int, default=0, help='Experiment ID')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=50, help='Size of embedding')
    parser.add_argument('--distance', type=str, default='hyperbolic', help='Type of distance to use')
    parser.add_argument('--center', type=str, default='geometric_median', help='')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--alpha', type=float, default=0.5, help='')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--std_noise', type=float, default=0.0, help='')
    parser.add_argument('--validation_multiple', action='store_true', default=False, help='')
    parser.add_argument('--evaluate_multiple', action='store_true', default=False, help='')
    parser.add_argument('--normalization', type=float, default=-1, help='')
    return parser

