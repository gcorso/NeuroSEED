from edit_distance.models.transformer.model import Transformer
from edit_distance.train import execute_train, general_arg_parser
from util.data_handling.data_loader import BOOL_CHOICE

parser = general_arg_parser()
parser.add_argument('--trans_layers', type=int, default=1, help='Number of attention layers')
parser.add_argument('--readout_layers', type=int, default=1, help='Number of final fully connected layers')
parser.add_argument('--hidden_size', type=int, default=16, help='Size of hidden tokens')
parser.add_argument('--mask', type=str, default="empty", help='Whether to apply a mask (empty or local{N})')
parser.add_argument('--heads', type=int, default=1, help='Number of attention heads at each layer')
parser.add_argument('--layer_norm', type=str, default='True', help='Layer normalization')
parser.add_argument('--segment_size', type=int, default=5, help='Number of elements for each string token')

args = parser.parse_args()

assert args.layer_norm in BOOL_CHOICE, "Boolean values have to be either 'True' or 'False' "

execute_train(model_class=Transformer,
              model_args=dict(segment_size=args.segment_size,
                              trans_layers=args.trans_layers,
                              readout_layers=args.readout_layers,
                              hidden_size=args.hidden_size,
                              mask=args.mask,
                              heads=args.heads,
                              layer_norm=True if args.layer_norm == 'True' else False),
              args=args)
