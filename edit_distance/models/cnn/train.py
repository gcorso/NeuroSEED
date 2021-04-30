from edit_distance.models.cnn.model import CNN
from edit_distance.train import execute_train, general_arg_parser
from util.data_handling.data_loader import BOOL_CHOICE

parser = general_arg_parser()
parser.add_argument('--readout_layers', type=int, default=1, help='Number of readout layers')
parser.add_argument('--channels', type=int, default=2, help='Number of channels at each convolutional layer')
parser.add_argument('--layers', type=int, default=5, help='Number of convolutional layers')
parser.add_argument('--stride', type=int, default=1, help='Stride in convolutions')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size in convolutions')
parser.add_argument('--pooling', type=str, default='avg', help='Pooling type (avg, max or none')
parser.add_argument('--non_linearity', type=str, default='True', help='Whether to apply non-linearity to convolutions')
parser.add_argument('--batch_norm', type=str, default='True', help='Batch normalization')
args = parser.parse_args()

assert args.non_linearity in BOOL_CHOICE and args.batch_norm in BOOL_CHOICE, \
    "Boolean values have to be either 'True' or 'False' "

execute_train(model_class=CNN,
              model_args=dict(readout_layers=args.readout_layers,
                              channels=args.channels,
                              layers=args.layers,
                              kernel_size=args.kernel_size,
                              pooling=args.pooling,
                              non_linearity=True if args.non_linearity == 'True' else False,
                              batch_norm=True if args.batch_norm == 'True' else False,
                              stride=args.stride),
              args=args)