"""
    Part of this code was adapted from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

"""Script to visualize the HypHCEmbeddings clustering."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_clustering.relaxed.datasets.hc_dataset import load_hc_data
from hierarchical_clustering.relaxed.models.linear import HypHCLinear
from util.ml_and_math.poincare import project
from hierarchical_clustering.relaxed.utils.visualization import plot_tree_from_leaves


def decode_tree_and_sequences(args, model, seq_loader, device):
    embedded_sequences = []
    for ids, seqs in seq_loader:
        e = model.encode(ids.to(device), seqs.to(device))
        embedded_sequences.append(e)

    embedded_sequences = torch.cat(embedded_sequences, dim=0)
    return model.decode_tree(embedded_sequences=embedded_sequences, fast_decoding=args.fast_decoding), embedded_sequences


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, default='./data',
                        help="path to a directory with a torch model_{seed}.pkl and a config.json files saved by train.py."
                        )
    parser.add_argument("--seed", type=str, default=1234, help="model seed to use")
    args = parser.parse_args()

    # load dataset
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    config_args = argparse.Namespace(**config)
    x, similarities = load_hc_data(config_args.dataset)
    print(x.shape)

    # build HypHCEmbeddings model
    model = HypHCLinear(temperature=config_args.temperature, rank=config_args.rank, init_size=config_args.init_size,
                        max_scale=config_args.max_scale, alphabet_size=config_args.alphabet_size,
                        sequence_length=x.shape[-1], device=device)
    params = torch.load(os.path.join(args.model_dir, f"model_{args.seed}.pkl"), map_location=torch.device('cpu'))
    model.load_state_dict(params, strict=False)
    model.to(device)
    model.eval()

    # get labels
    with open('./data/animals.fasta', 'rb') as f:
        L = f.readlines()
    labels = [l[2:-2].decode('UTF-8') for i, l in enumerate(L) if i%2 == 0]

    # decode tree
    seq_loader = DataLoader(TensorDataset(torch.arange(x.shape[0]), torch.from_numpy(x)), batch_size=config_args.batch_size, shuffle=False)
    tree, leaves_embeddings = decode_tree_and_sequences(args=config_args, model=model, seq_loader=seq_loader, device=device)
    leaves_embeddings = model.normalize_embeddings(leaves_embeddings)
    leaves_embeddings = project(leaves_embeddings).detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax = plot_tree_from_leaves(ax, tree, leaves_embeddings, labels=labels)
    plt.show()
    fig.savefig(os.path.join(args.model_dir, f"embeddings_{args.seed}.png"))

