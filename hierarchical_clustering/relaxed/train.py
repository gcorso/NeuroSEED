"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

"""Train a hyperbolic embedding model for hierarchical clustering."""

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from hierarchical_clustering.relaxed import optim
from hierarchical_clustering.relaxed.config import config_args
from hierarchical_clustering.relaxed.datasets.hc_dataset import HCTripletDataset, load_hc_data
from hierarchical_clustering.relaxed.models.cnn import HypHCCNN
from hierarchical_clustering.relaxed.models.embeddings import HypHCEmbeddings
from hierarchical_clustering.relaxed.models.linear import HypHCLinear
from hierarchical_clustering.relaxed.models.mlp import HypHCMLP
from hierarchical_clustering.relaxed.utils.metrics import dasgupta_cost
from hierarchical_clustering.relaxed.utils.training import add_flags_from_config, get_savedir


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get saving directory
    if args.save:
        save_dir = get_savedir(args)
        logging.info("Save directory: " + save_dir)
        save_path = os.path.join(save_dir, "model_{}.pkl".format(args.seed))
        if False and os.path.exists(save_dir):
            if os.path.exists(save_path):
                logging.info("Model with the same configuration parameters already exists.")
                logging.info("Exiting")
                return
        else:
            #os.makedirs(save_dir)
            with open(os.path.join(save_dir, "config.json"), 'w') as fp:
                json.dump(args.__dict__, fp)

    # set seed
    print("Using seed {}.".format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set precision
    print("Using {} precision.".format(args.dtype))
    if args.dtype == "double":
        torch.set_default_dtype(torch.float64)

    # create dataset
    x, similarities = load_hc_data(args.dataset)
    dataset = HCTripletDataset(x, similarities, num_samples=args.num_samples)
    triplet_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    seq_loader = DataLoader(TensorDataset(torch.arange(x.shape[0]), torch.from_numpy(x)), batch_size=args.batch_size,
                            shuffle=False)

    # create model
    if args.model == 'embeddings':
        model = HypHCEmbeddings(n_nodes=dataset.n_nodes, rank=args.rank, temperature=args.temperature,
                                init_size=args.init_size, max_scale=args.max_scale)
    elif args.model == 'linear':
        model = HypHCLinear(rank=args.rank, temperature=args.temperature, init_size=args.init_size,
                            max_scale=args.max_scale, alphabet_size=args.alphabet_size,
                            sequence_length=dataset.len_sequences, device=device)
    elif args.model == 'mlp':
        model = HypHCMLP(rank=args.rank, temperature=args.temperature, init_size=args.init_size,
                         max_scale=args.max_scale, alphabet_size=args.alphabet_size,
                         sequence_length=dataset.len_sequences, device=device,
                         layers=args.layers, hidden_size=args.hidden_size, dropout=args.dropout,
                         batch_norm=args.batch_norm)
    elif args.model == 'cnn':
        model = HypHCCNN(rank=args.rank, temperature=args.temperature, init_size=args.init_size,
                         max_scale=args.max_scale, alphabet_size=args.alphabet_size,
                         sequence_length=dataset.len_sequences, device=device,
                         layers=args.layers, channels=args.channels, kernel_size=args.kernel_size,
                         readout_layers=args.readout_layers, non_linearity=args.non_linearity, pooling=args.pooling,
                         dropout=args.dropout, batch_norm=args.batch_norm)
    else:
        raise ValueError

    model.to(device)

    # create optimizer
    Optimizer = getattr(optim, args.optimizer)
    optimizer = Optimizer(model.parameters(), args.learning_rate)

    # train model
    best_cost = np.inf
    best_model = None
    counter = 0
    print("Start training")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, (triple_ids, triple_sequences, triple_similarities) in enumerate(triplet_loader):
            triple_ids, triple_sequences, triple_similarities = \
                triple_ids.to(device), triple_sequences.to(device), triple_similarities.to(device)
            loss = model.loss(triple_ids, triple_sequences, triple_similarities)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model.scale.data[0] < model.init_size:
                model.scale.data[0] = model.init_size

            total_loss += loss
        total_loss = total_loss.item() / (step + 1.0)
        print("Epoch {} | average train loss: {:.6f} scale {:.6f}".format(epoch, total_loss, model.scale.data[0]))

        # keep best embeddings
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            tree = decode_tree(args, model, seq_loader, device)
            cost = dasgupta_cost(tree, similarities)
            print("Current Dasgupta's cost: {:.4f}".format(cost))
            if cost < best_cost:
                counter = 0
                best_cost = cost
                best_model = model.state_dict()
            else:
                counter += 1
                if counter == args.patience:
                    print("Early stopping.")
                    break

        # anneal temperature
        if (epoch + 1) % args.anneal_every == 0:
            model.anneal_temperature(args.anneal_factor)
            print("Annealing temperature to: {}".format(model.temperature))
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_factor
                lr = param_group['lr']
            print("Annealing learning rate to: {}".format(lr))

    print("Optimization finished.")
    if args.save and best_model is not None:
        # save best embeddings
        model.load_state_dict(best_model)
        print("Saving best model at {}".format(save_path))
        torch.save(best_model, save_path)

    # evaluation
    print("Final Dasgupta's cost: {:.4f}".format(best_cost))
    return


def decode_tree(args, model, seq_loader, device):
    embedded_sequences = []
    for ids, seqs in seq_loader:
        e = model.encode(ids.to(device), seqs.to(device))
        embedded_sequences.append(e)

    embedded_sequences = torch.cat(embedded_sequences, dim=0)
    return model.decode_tree(embedded_sequences=embedded_sequences, fast_decoding=args.fast_decoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser = add_flags_from_config(parser, config_args)
    args = parser.parse_args()
    train(args)
