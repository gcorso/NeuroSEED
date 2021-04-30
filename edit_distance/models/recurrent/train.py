from edit_distance.models.recurrent.model import GRU
from edit_distance.train import execute_train, general_arg_parser

parser = general_arg_parser()
parser.add_argument('--recurrent_layers', type=int, default=1, help='Number of recurrent layers')
parser.add_argument('--readout_layers', type=int, default=2, help='Number of readout layers')
parser.add_argument('--hidden_size', type=int, default=100, help='Size of hidden dimension')
args = parser.parse_args()

execute_train(model_class=GRU,
              model_args=dict(recurrent_layers=args.recurrent_layers,
                              readout_layers=args.readout_layers,
                              hidden_size=args.hidden_size),
              args=args)
