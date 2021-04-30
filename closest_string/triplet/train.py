import argparse
import os
import pickle
import sys
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from closest_string.test import closest_string_testing
from closest_string.triplet.dataset import TripletDataset
from closest_string.triplet.models.triplet_encoder import TripletEncoder, triplet_loss
from edit_distance.models.hyperbolics import RAdam
from hierarchical_clustering.unsupervised.unsupervised import hierarchical_clustering_testing
from util.data_handling.data_loader import get_dataloaders
from util.ml_and_math.loss_functions import AverageMeter


def general_arg_parser():
    """ Parsing of parameters common to all the different models """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Dataset path')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training (GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--print_every', type=int, default=1, help='Print training results every')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=16, help='Size of embedding')
    parser.add_argument('--distance', type=str, default='euclidean', help='Type of distance to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--loss', type=str, default="mse", help='Loss function to use (mse, mape or mae)')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot real vs predicted distances')
    parser.add_argument('--closest_data_path', type=str, default='', help='Dataset for closest string retrieval tests')
    parser.add_argument('--hierarchical_data_path', type=str, default='', help='Dataset for hierarchical clustering')
    parser.add_argument('--scaling', type=str, default='False', help='Project to hypersphere (for hyperbolic)')
    parser.add_argument('--hyp_optimizer', type=str, default='Adam', help='Optimizer for hyperbolic (Adam or RAdam)')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin loss')

    return parser


def execute_train(model_class, model_args, args):
    # set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'
    print('Using device:', device)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    datasets = load_triplet_dataset(args.data)
    loaders = get_dataloaders(datasets, batch_size=args.batch_size, workers=args.workers)

    # fix hyperparameters
    model_args = SimpleNamespace(**model_args)
    model_args.device = device
    model_args.len_sequence = datasets['train'].len_sequence
    model_args.embedding_size = args.embedding_size
    model_args.dropout = args.dropout
    print("Length of sequence", datasets['train'].len_sequence)
    args.scaling = True if args.scaling == 'True' else False

    # generate model
    embedding_model = model_class(**vars(model_args))
    model = TripletEncoder(embedding_model=embedding_model, distance=args.distance, scaling=args.scaling)
    model.to(device)

    # select optimizer
    if args.distance == 'hyperbolic' and args.hyp_optimizer == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # select loss
    loss = partial(triplet_loss, margin=args.margin, device=device)

    # print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params', total_params)

    # Train model
    t_total = time.time()
    bad_counter = 0
    best = 1e10
    best_epoch = -1
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        loss_train = train(model, loaders['train'], optimizer, loss, device)
        loss_val = test(model, loaders['val'], loss, device)

        # print progress
        if epoch % args.print_every == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.6f}'.format(loss_train),
                  'loss_val: {:.6f}'.format(loss_val),
                  'time: {:.4f}s'.format(time.time() - t))
            sys.stdout.flush()

        if loss_val < best:
            # save current model
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            # remove previous model
            if best_epoch >= 0:
                os.remove('{}.pkl'.format(best_epoch))
            # update training variables
            best = loss_val
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            print('Early stop at epoch {} (no improvement in last {} epochs)'.format(epoch + 1, bad_counter))
            break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch + 1))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    for dset in loaders.keys():
        avg_loss = test(model, loaders[dset], loss, device)
        print('Final results {}: loss = {:.6f}'.format(dset, avg_loss))


    # Nearest neighbour retrieval
    if args.closest_data_path != '':
        print("Closest string retrieval")
        closest_string_testing(encoder_model=model, data_path=args.closest_data_path,
                               batch_size=args.batch_size, device=device, distance=args.distance)

    # Hierarchical clustering
    if args.hierarchical_data_path != '':
        print("Hierarchical clustering")
        hierarchical_clustering_testing(encoder_model=model, data_path=args.hierarchical_data_path,
                                        batch_size=args.batch_size, device=device, distance=args.distance)

    #torch.save((model_class, model_args, model.embedding_model.state_dict(), args.distance), '{}.pkl'.format(model_class.__name__))


def load_triplet_dataset(path):
    with open(path, 'rb') as f:
        sequences, distances = pickle.load(f)

    datasets = {}
    for key in sequences.keys():
        datasets[key] = TripletDataset(sequences[key].unsqueeze(0), distances[key].unsqueeze(0), multiplicity=10)
    return datasets


def train(model, loader, optimizer, loss, device):
    avg_loss = AverageMeter()
    model.train()

    for sequences in loader:
        # move examples to right device
        sequences = sequences.to(device)

        # forward propagation
        optimizer.zero_grad()
        distances = model(sequences)

        # loss and backpropagation
        loss_train = loss(*distances)
        loss_train.backward()
        optimizer.step()

        # keep track of average loss
        avg_loss.update(loss_train.data.item(), sequences.shape[0])

    return avg_loss.avg


def test(model, loader, loss, device):
    avg_loss = AverageMeter()
    model.eval()

    for sequences in loader:
        # move examples to right device
        sequences = sequences.to(device)

        # forward propagation and loss computation
        distances = model(sequences)
        loss_val = loss(*distances).data.item()
        avg_loss.update(loss_val, sequences.shape[0])

    return avg_loss.avg
