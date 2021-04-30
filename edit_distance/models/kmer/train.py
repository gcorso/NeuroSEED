from edit_distance.models.kmer.model import Kmer
from edit_distance.train import execute_train, general_arg_parser

parser = general_arg_parser()
parser.add_argument('--k', type=int, default=4, help='Size of kernel')

args = parser.parse_args()


execute_train(model_class=Kmer,
              model_args=dict(k=args.k),
              args=args)
