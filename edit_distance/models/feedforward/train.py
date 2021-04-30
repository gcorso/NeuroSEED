from edit_distance.models.feedforward.model import MLPEncoder
from edit_distance.train import execute_train, general_arg_parser
from util.data_handling.data_loader import BOOL_CHOICE

parser = general_arg_parser()
parser.add_argument('--layers', type=int, default=1, help='Number of fully connected layers')
parser.add_argument('--hidden_size', type=int, default=100, help='Size of hidden layers')
parser.add_argument('--batch_norm', type=str, default='False', help='Batch normalization')

args = parser.parse_args()

assert args.batch_norm in BOOL_CHOICE, "Boolean values have to be either 'True' or 'False' "

execute_train(model_class=MLPEncoder,
              model_args=dict(layers=args.layers,
                              hidden_size=args.hidden_size,
                              batch_norm=True if args.batch_norm == 'True' else False),
              args=args)
