import argparse
# from easydict import EasyDict as edict
import json

parser = argparse.ArgumentParser(description="DESAH demo")
parser.add_argument('--bits', default='32', type=str,help='binary code length (default: 128)')
parser.add_argument('--gpu', default='0', type=str,help='selected gpu (default: 0)')
parser.add_argument('--BETA', default=0.8, type=float, help='hyper-parameter: balance parameter')
parser.add_argument('--batch-size', default=16, type=int, help='batch size (default: 32)')
parser.add_argument('--CODE_LEN', default=32, type=int, help='binary code length (default: 16)')
parser.add_argument('--LAMBDA1', default=0.3, type=float, help='hyper-parameter:  (default: )')
parser.add_argument('--LAMBDA2', default=0.3, type=float, help='hyper-parameter: (default: )')
parser.add_argument('--LAMBDA3', default=0.01, type=float, help='hyper-parameter:  (default: )')
parser.add_argument('--LAMBDA4', default=40, type=float, help='hyper-parameter: (default: )')
parser.add_argument('--LAMBDA5', default=40, type=float, help='hyper-parameter:  (default: )')
parser.add_argument('--kx', default=2, type=float,help='hyper-parameter: balance parameter')
parser.add_argument('--ky', default=0.2, type=float, help='hyper-parameter: balance parameter')
parser.add_argument('--NUM-EPOCH', default=200, type=int, help='hyper-parameter: EPOCH (default: 40)')
parser.add_argument('--LR-IMG', default=0.001, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-TXT', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-2)')
parser.add_argument('--MOMENTUM', default=0.9, type=float, help='hyper-parameter: momentum (default: 0.9)')
parser.add_argument('--WEIGHT-DECAY', default=5e-4, type=float, help='hyper-parameter: weight decay (default: )')
parser.add_argument('--NUM-WORKERS', default=4, type=int, help='workers (default: )')
parser.add_argument('--EVAL', default= False, type=bool,help='')
parser.add_argument('--EPOCH-INTERVAL', default=1, type=int, help='INTERVAL (default: 2)')
parser.add_argument('--EVAL-INTERVAL', default=5, type=int, help='evaluation interval (default:)')

args = parser.parse_args()

# load basic settings
# with open(args.Config, 'r') as f:
    # config = edict(json.load(f))

# update settings
# config.TRAIN = args.Train
# config.DATASET = args.Dataset