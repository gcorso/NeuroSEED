from multiple_alignment.steiner_string.models.convolutional.model import CNNEncoder, CNNDecoder
from multiple_alignment.steiner_string.train import execute_train
from multiple_alignment.steiner_string.parser import general_arg_parser

parser = general_arg_parser()
parser.add_argument('--readout_layers', type=int, default=2, help='')
parser.add_argument('--channels', type=int, default=20, help='')
parser.add_argument('--layers', type=int, default=2, help='')
parser.add_argument('--kernel_size', type=int, default=3, help='')
parser.add_argument('--non_linearity', type=bool, default=False, help='')
args = parser.parse_args()

execute_train(encoder_class=CNNEncoder, decoder_class=CNNDecoder,
              encoder_args=dict(readout_layers=args.readout_layers,
                                channels=args.channels,
                                layers=args.layers,
                                kernel_size=args.kernel_size,
                                non_linearity=args.non_linearity),
              decoder_args=dict(readout_layers=args.readout_layers,
                                channels=args.channels,
                                layers=args.layers,
                                kernel_size=args.kernel_size,
                                non_linearity=args.non_linearity),
              args=args)
