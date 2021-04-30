import os
import pickle
import sys
import time
from types import SimpleNamespace
from functools import partial
import numpy as np
import torch
import torch.optim as optim

from multiple_alignment.steiner_string.task.dataset import PairDistanceDatasetSampled, MultipleAlignmentDataset
from multiple_alignment.steiner_string.models.pair_autoencoder import PairDistanceAutoEncoder
from multiple_alignment.steiner_string.models.loss import pair_autoencoder_loss, multiple_alignment_cost
from multiple_alignment.steiner_string.models.msa_autoencoder import MultipleAlignmentAutoEncoder
from util.data_handling.data_loader import get_dataloaders
from util.ml_and_math.loss_functions import AverageMeter


def execute_train(encoder_class, decoder_class, encoder_args, decoder_args, args, autoregressive=False):
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
    datasets = load_msa_distance_dataset(args.data)
    loaders = get_dataloaders(datasets, batch_size=args.batch_size, workers=args.workers)

    # add parameters
    additional_args = {
        'device': device,
        'len_sequence': datasets['train'].len_sequence,
        'embedding_size': args.embedding_size,
        'dropout': args.dropout
    }
    encoder_args = SimpleNamespace(**encoder_args, **additional_args)
    decoder_args = SimpleNamespace(**decoder_args, **additional_args)

    # create models
    encoder = encoder_class(**vars(encoder_args))
    decoder = decoder_class(**vars(decoder_args))
    pair_autoencoder = PairDistanceAutoEncoder(encoder=encoder, decoder=decoder, distance=args.distance, device=device,
                                               std_noise=args.std_noise, normalization=args.normalization, autoregressive=autoregressive)
    multiple_autoencoder = MultipleAlignmentAutoEncoder(encoder=encoder, decoder=decoder, center=args.center,
                                                        device=device, normalization=args.normalization,
                                                        distance=args.distance)

    # create optimizer and loss
    optimizer = optim.Adam(pair_autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = partial(pair_autoencoder_loss, alpha=args.alpha)

    # print total number of parameters
    total_params = sum(p.numel() for p in pair_autoencoder.parameters() if p.requires_grad)
    print('Total params', total_params)

    # Train model
    t_total = time.time()
    bad_counter = 0
    best = 1e10
    best_epoch = -1
    start_epoch = 0
    best_losses = 0

    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        loss_train, subdiv_train = train(pair_autoencoder, loaders['train'], optimizer, loss, device)
        loss_val, subdiv_val = test_pair(pair_autoencoder, loaders['val'], loss, device)

        # print progress
        if epoch % args.print_every == 0:
            if args.evaluate_multiple:
                cost_multiple, cost_expected = test_multiple(multiple_autoencoder, loaders['test'], device,
                                                             datasets['train'].normalisation_constant)
            else:
                cost_multiple, cost_expected = 0, 0
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.6f} ({:.6f}, {:.6f}, {:.6f}, {:.6f})'.format(loss_train, *subdiv_train),
                  'loss_val: {:.6f} ({:.6f}, {:.6f}, {:.6f}, {:.6f})'.format(loss_val, *subdiv_val),
                  'test_multiple: {:.4f} {:.4f}'.format(cost_multiple, cost_expected),
                  'time: {:.4f}s'.format(time.time() - t))
            sys.stdout.flush()

        if epoch % args.eval_every == 0:

            if args.validation_multiple:
                multiple_val, _ = test_multiple(multiple_autoencoder, loaders['val_msa'], device, datasets['train'].normalisation_constant)

                print('Epoch: {:04d}'.format(epoch + 1),
                      'val_multiple: {:.4f}'.format(multiple_val),
                      'time: {:.4f}s'.format(time.time() - t))

            if (multiple_val if args.validation_multiple else loss_val) < best:
                # save current model
                torch.save(pair_autoencoder.state_dict(), '{}.pkl'.format(epoch))
                # remove previous model
                if best_epoch >= 0:
                    os.remove('{}.pkl'.format(best_epoch))
                # update training variables
                best = (multiple_val if args.validation_multiple else loss_val)
                best_losses = (loss_train, subdiv_train, loss_val, subdiv_val)
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
    pair_autoencoder.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    print('Final loss_train: {:.6f} ({:.6f}, {:.6f}, {:.6f}, {:.6f})'.format(best_losses[0], *best_losses[1]))
    print('Final loss_val: {:.6f} ({:.6f}, {:.6f}, {:.6f}, {:.6f})'.format(best_losses[2], *best_losses[3]))
    print('Final multiple val: {:.4f} '.format(test_multiple(multiple_autoencoder, loaders['val_msa'], device, datasets['train'].normalisation_constant)[0]))
    cost_multiple, cost_expected = test_multiple(multiple_autoencoder, loaders['test'], device, datasets['train'].normalisation_constant)
    print("Cost multiple", cost_multiple, cost_expected)


def load_msa_distance_dataset(path):
    with open(path, 'rb') as f:
        datasets_raw = pickle.load(f)

    datasets = {}
    datasets['train'] = PairDistanceDatasetSampled(datasets_raw['train']['texts'].unsqueeze(0),
                                                   datasets_raw['train']['distances'].unsqueeze(0), multiplicity=10)
    datasets['val'] = PairDistanceDatasetSampled(datasets_raw['val']['texts'].unsqueeze(0),
                                                 datasets_raw['val']['distances'].unsqueeze(0), multiplicity=10)
    datasets['val_msa'] = MultipleAlignmentDataset(datasets_raw['val_msa']['texts'])
    datasets['test'] = MultipleAlignmentDataset(datasets_raw['test']['texts'])
    return datasets


def train(pair_autoencoder, loader, optimizer, loss, device):
    avg_loss = AverageMeter()
    avg_sub_div = AverageMeter(len_tuple=4)
    pair_autoencoder.train()

    for sequences, distances in loader:
        sequences, distances = sequences.to(device), distances.to(device)
        optimizer.zero_grad()
        predicted_distance, reconstructed_sequences = pair_autoencoder(sequences)
        loss_train, sub_div = loss(reconstructed_sequences, predicted_distance, sequences, distances)
        loss_train.backward()
        optimizer.step()
        avg_loss.update(loss_train, sequences.shape[0])
        avg_sub_div.update(sub_div, sequences.shape[0])

    return avg_loss.avg, avg_sub_div.avg


def test_pair(pair_autoencoder, loader, loss, device):
    avg_loss = AverageMeter()
    avg_sub_div = AverageMeter(len_tuple=4)
    pair_autoencoder.eval()

    for sequences, distances in loader:
        sequences, distances = sequences.to(device), distances.to(device)
        predicted_distance, reconstructed_sequences = pair_autoencoder(sequences)
        loss_val, sub_div = loss(reconstructed_sequences, predicted_distance, sequences, distances)
        avg_loss.update(loss_val, sequences.shape[0])
        avg_sub_div.update(sub_div, sequences.shape[0])

    return avg_loss.avg, avg_sub_div.avg


def test_multiple(multiple_autoencoder, loader, device, normalisation_constant=1.0):
    avg_cost = AverageMeter()
    avg_exp_cost = AverageMeter()
    multiple_autoencoder.eval()

    for sequences in loader:
        sequences = sequences.to(device)
        centers, avg_exp = multiple_autoencoder(sequences)
        cost = multiple_alignment_cost(sequences, centers)
        avg_cost.update(cost, sequences.shape[0])
        avg_exp_cost.update(avg_exp, sequences.shape[0])

    return avg_cost.avg, avg_exp_cost.avg * normalisation_constant
