from functools import partial

import numpy as np

from edit_distance.baselines.train import general_arg_parser, test_method


def kmer(S, k, alphabet_size=4, compress_zeros=False):
    kernel = [alphabet_size**p for p in range(k)]
    kmers = np.apply_along_axis(partial(np.convolve, v=kernel, mode='valid'), 1, S)

    if compress_zeros:
        # if compress_zeros = True subsequences not contained in any of the sequences are ignored
        x, indices = np.unique(kmers, return_inverse=True)
        kmers = indices.reshape(kmers.shape)
        dimensions = len(x)
        vectors = np.zeros((S.shape[0], dimensions))
    else:
        vectors = np.zeros((S.shape[0], alphabet_size**k))

    for d in range(len(S)):
        bbins = np.bincount(kmers[d])
        vectors[d][:len(bbins)] += bbins

    return vectors


parser = general_arg_parser()
args = parser.parse_args()

test_method(encoder_methods={
                'k2': partial(kmer, k=2),
                'k3': partial(kmer, k=3),
                'k4': partial(kmer, k=4),
                'k5': partial(kmer, k=5),
                'k6': partial(kmer, k=6),
            },
            args=args)