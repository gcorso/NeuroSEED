"""Configuration parameters."""

config_args = {
    # training
    "seed": 1234,
    "epochs": 10,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "eval_every": 5,
    "patience": 10,
    "optimizer": "RAdam",
    "save": 1,
    "fast_decoding": 1,
    "num_samples": 50,

    # model
    "model": "linear",
    "dtype": "double",
    "rank": 2,
    "temperature": 0.01,
    "init_size": 1e-2,
    "anneal_every": 20,
    "anneal_factor": 1.0,
    "max_scale": 1 - 1e-3,

    "layers": 2,
    "hidden_size": 100,
    "dropout": 0.0,
    "batch_norm": False,

    "channels": 32,
    "kernel_size": 3,
    "readout_layers": 2,
    "non_linearity": True,
    "pooling": 'avg',

    # dataset
    "dataset": "",
    "alphabet_size": 4,
}
