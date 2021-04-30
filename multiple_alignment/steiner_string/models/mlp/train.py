from multiple_alignment.steiner_string.models.mlp.model import MLPEncoder, MLPDecoder
from multiple_alignment.steiner_string.train import execute_train
from multiple_alignment.steiner_string.parser import general_arg_parser

parser = general_arg_parser()
parser.add_argument('--hidden_size', type=int, default=10, help='')
parser.add_argument('--layers', type=int, default=1, help='')
args = parser.parse_args()

execute_train(encoder_class=MLPEncoder, decoder_class=MLPDecoder,
              encoder_args=dict(hidden_size=args.hidden_size,
                                layers=args.layers),
              decoder_args=dict(hidden_size=args.hidden_size,
                                layers=args.layers),
              args=args)
