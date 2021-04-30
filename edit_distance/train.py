import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from closest_string.test import closest_string_testing
from edit_distance.task.dataset import EditDistanceDatasetSampled, EditDistanceDatasetComplete
from edit_distance.models.hyperbolics import RAdam
from edit_distance.models.pair_encoder import PairEmbeddingDistance
from hierarchical_clustering.unsupervised.unsupervised import hierarchical_clustering_testing
from multiple_alignment.guide_tree.guide_tree import approximate_guide_trees
from util.data_handling.data_loader import get_dataloaders
from util.ml_and_math.loss_functions import MAPE
from util.ml_and_math.loss_functions import AverageMeter


def general_arg_parser():
    """ Parsing of parameters common to all the different models """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../../data/edit_qiita_small.pkl', help='Dataset path')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training (GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--patience', type=int, default=50, help='Patience')
    parser.add_argument('--print_every', type=int, default=1, help='Print training results every')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=5, help='Size of embedding')
    parser.add_argument('--distance', type=str, default='hyperbolic', help='Type of distance to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--loss', type=str, default="mse", help='Loss function to use (mse, mape or mae)')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot real vs predicted distances')
    parser.add_argument('--closest_data_path', type=str, default='', help='Dataset for closest string retrieval tests')
    parser.add_argument('--hierarchical_data_path', type=str, default='', help='Dataset for hierarchical clustering')
    parser.add_argument('--construct_msa_tree', type=str, default='False', help='Whether to construct NJ tree testset')
    parser.add_argument('--extr_data_path', type=str, default='', help='Dataset for further edit distance tests')
    parser.add_argument('--scaling', type=str, default='False', help='Project to hypersphere (for hyperbolic)')
    parser.add_argument('--hyp_optimizer', type=str, default='Adam', help='Optimizer for hyperbolic (Adam or RAdam)')
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
    datasets = load_edit_distance_dataset(args.data)
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
    model = PairEmbeddingDistance(embedding_model=embedding_model, distance=args.distance, scaling=args.scaling)
    model.to(device)

    # select optimizer
    if args.distance == 'hyperbolic' and args.hyp_optimizer == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # select loss
    loss = None
    if args.loss == "mse":
        loss = nn.MSELoss()
    elif args.loss == "mae":
        loss = nn.L1Loss()
    elif args.loss == "mape":
        loss = MAPE

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
                  'loss_val: {:.6f} MAPE {:.4f}'.format(*loss_val),
                  'time: {:.4f}s'.format(time.time() - t))
            sys.stdout.flush()

        if loss_val[0] < best:
            # save current model
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            # remove previous model
            if best_epoch >= 0:
                os.remove('{}.pkl'.format(best_epoch))
            # update training variables
            best = loss_val[0]
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
        if args.plot:
            avg_loss = test_and_plot(model, loaders[dset], loss, device, dset)
        else:
            avg_loss = test(model, loaders[dset], loss, device)
        print('Final results {}: loss = {:.6f}  MAPE {:.4f}'.format(dset, *avg_loss))

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

    # MSA tree construction on test set
    if args.construct_msa_tree == 'True':
        print("MSA tree construction")
        approximate_guide_trees(encoder_model=model, dataset=datasets['test'],
                                batch_size=args.batch_size, device=device, distance=args.distance)

    # Extra datasets testing (e.g. extrapolation)
    if args.extr_data_path != '':
        print("Extra datasets testing")
        datasets = load_edit_distance_dataset(args.extr_data_path)
        loaders = get_dataloaders(datasets, batch_size=max(1, args.batch_size // 8), workers=args.workers)

        for dset in loaders.keys():
            if args.plot:
                avg_loss = test_and_plot(model, loaders[dset], loss, device, dset)
            else:
                avg_loss = test(model, loaders[dset], loss, device)
            print('Final results {}: loss = {:.6f}  MAPE {:.4f}'.format(dset, *avg_loss))

    torch.save((model_class, model_args, model.embedding_model.state_dict(), args.distance),
               '{}.pkl'.format(model_class.__name__))


def load_edit_distance_dataset(path):
    with open(path, 'rb') as f:
        sequences, distances = pickle.load(f)

    datasets = {}
    for key in sequences.keys():
        if len(sequences[key].shape) == 2:  # datasets without batches
            if key == 'train':
                datasets[key] = EditDistanceDatasetSampled(sequences[key].unsqueeze(0), distances[key].unsqueeze(0),
                                                           multiplicity=10)
            else:
                datasets[key] = EditDistanceDatasetComplete(sequences[key], distances[key])
        else:  # datasets with batches
            datasets[key] = EditDistanceDatasetSampled(sequences[key], distances[key])
    return datasets


def train(model, loader, optimizer, loss, device):
    avg_loss = AverageMeter()
    model.train()

    for sequences, labels in loader:
        # move examples to right device
        sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation
        optimizer.zero_grad()
        output = model(sequences)

        # loss and backpropagation
        loss_train = loss(output, labels)
        loss_train.backward()
        optimizer.step()

        # keep track of average loss
        avg_loss.update(loss_train.data.item(), sequences.shape[0])

    return avg_loss.avg


def test(model, loader, loss, device):
    avg_loss = AverageMeter(len_tuple=2)
    model.eval()

    for sequences, labels in loader:
        # move examples to right device
        sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation and loss computation
        output = model(sequences)
        loss_val = loss(output, labels).data.item()
        mape = MAPE(output, labels).data.item()
        avg_loss.update((loss_val, mape), sequences.shape[0])

    return avg_loss.avg


def test_and_plot(model, loader, loss, device, dataset):
    avg_loss = AverageMeter(len_tuple=2)
    model.eval()

    output_list = []
    labels_list = []

    for sequences, labels in loader:
        # move examples to right device
        sequences, labels = sequences.to(device), labels.to(device)

        # forward propagation and loss computation
        output = model(sequences)
        loss_val = loss(output, labels).data.item()
        mape = MAPE(output, labels).data.item()
        avg_loss.update((loss_val, mape), sequences.shape[0])

        # append real and predicted distances to lists
        output_list.append(output.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    # save real and predicted distances for offline plotting
    outputs = np.concatenate(output_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    pickle.dump((outputs, labels), open(dataset + ".pkl", "wb"))
    # plt.plot(outputs, labels, 'o', color='black')
    # plt.show()

    return avg_loss.avg
