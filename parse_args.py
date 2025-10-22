import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                        type=str,
                        default = 'PrimeKG',
                        help="dataset for training")

argparser.add_argument('--seed', type=int, default=1234, help='random seed')

argparser.add_argument('--convE_channels', type=int, default=32,help='convE_channels')

argparser.add_argument('--convE_kernel_size', type=int, default=3,help='convE_kernel_size')

argparser.add_argument('--num_layers_encoder', type=int, default=3, help='number of layers')

argparser.add_argument('--gpu', type=int, default=0, help='gpu device')

argparser.add_argument('--freeze_pretrained', type=bool, default=False, help='whether to freeze pretrained embeddings')

argparser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

argparser.add_argument('--epochs', type=int, default=1000, help='training epoch')

argparser.add_argument('--batch_size', type=int, default=256, help='batch size')

argparser.add_argument('--patience', type=int, default = 50, help='early stopping patience')

argparser.add_argument('--weight_decay', type=float, default=1e-3, help='AdamW weight decay')

argparser.add_argument('--K_fold', type=int, default=10, help='k-fold cross validation')

argparser.add_argument('--validation_ratio', type=float, default=0.1, help='validation set partition ratio')

argparser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')

argparser.add_argument('--in_dim', default=128, type=int, help='input dimension')

argparser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension')

argparser.add_argument('--out_dim', default=64, type=int, help='output dimension')

argparser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')

argparser.add_argument('--dropout', default=0.2, type=float, help='dropout')

argparser.add_argument('--mask_strategy', type=str, default='fixed', help='masking strategy: curriculum or fixed')
argparser.add_argument('--min_mask_rate', type=float, default=0.1, help='minimum mask rate for curriculum strategy')
argparser.add_argument('--max_mask_rate', type=float, default=0.9, help='maximum mask rate for curriculum strategy')
argparser.add_argument('--fixed_mask_rate', type=float, default=0.5, help='fixed mask rate if using fixed strategy')
argparser.add_argument('--remask_rate', type=float, default=0.5, help='mask rate during training for re-masking')
argparser.add_argument('--remask_view', type=int, default=3, help='number of re-masked views')
argparser.add_argument('--scale_factor', type=float, default=3.0, help='scaled cosine error loss factor')
argparser.add_argument('--alpha', type=float, default=0.5, help='weighting factor for reconstruction loss')

argparser.add_argument('--num_layers_decoder', default=2, type=int, help='number of layers')

args = argparser.parse_args()