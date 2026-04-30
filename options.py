import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs to train.')
parser.add_argument('--lr1', type=float, default=0.0001, help='learning rate of generator.')
parser.add_argument('--lr2', type=float, default=0.0001, help='learning rate of discriminator.')
parser.add_argument('--d_lr_factor', type=float, default=0.2, help='Factor to reduce Discriminator LR relative to lr2.')
parser.add_argument('--batch_size', type=int, default=2, help='batchsize.')
parser.add_argument('--patch_size', type=int, default=256, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers.')

parser.add_argument('--t_weight', type=float, default=5.0, help='weights of t loss.')
parser.add_argument('--adv_weight', type=float, default=1.0, help='weights of adversial loss.')
parser.add_argument('--perceptual_weight', type=float, default=0.1, help='weights of perceptual loss.')
parser.add_argument('--lambda_continuity', type=float, default=10.0, help='weight for continuity loss.')
parser.add_argument('--tv_weight', type=float, default=0.5, help='weight for total variation loss.')
parser.add_argument('--color_weight', type=float, default=5.0, help='weight for color consistency loss.')

parser.add_argument('--max_beta', type=float, default=3.5, help='Maximum beta value.')
parser.add_argument('--uniform_haze_strength', type=float, default=0.5, help='Strength of haze on foreground (0.0-1.0).')
parser.add_argument('--prob_max_haze', type=float, default=0.4, help='Probability of forcing alpha=1.0.')

parser.add_argument('--real_label_val', type=float, default=0.9, help='Label smoothing value for real images.')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping.')

parser.add_argument('--indoor_path', type=str, default="/workspace/Haze2610/n2h/train/A/", help='indoor clean images')
parser.add_argument('--outdoor_path', type=str, default="/workspace/Haze2610/n2h/train/A/", help='outdoor clean images')
parser.add_argument('--rw_path', type=str, default="/workspace/Haze2610/n2h/train/B/", help='real world hazy images')
parser.add_argument('--output_path', type=str, default="datasets/output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint saving path')
parser.add_argument('--use-swin-discriminator', action='store_true', help='use Swin Transformer for discriminator')

options = parser.parse_args()