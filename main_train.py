# -----------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Main module for training the SSH network on a given dataset
# Written by Mahyar Najibi
# -----------------------------------------------------------

from SSH.train import train_net, get_training_roidb
import argparse
import numpy as np
from datasets.factory import get_imdb
from utils.get_config import cfg, cfg_from_file, cfg_from_list, get_output_dir, cfg_print


def parser():
    parser = argparse.ArgumentParser('SSH Train Module!',
                            description='You can change other configs by providing a YAML config file!')
    parser.add_argument('--db', dest='db_name', help='Path to the image',
                        default='wider_train', type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='The GPU id[s] to be used',
                        default='0,1,2,3', type=str)
    parser.add_argument('--solver', dest='solver_proto', help='SSH caffe solver prototxt',
                        default='SSH/models/solver_ssh.prototxt', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Output path for saving the figure',
                        default='data/demo', type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='Pre-trained model',
                        default='data/imagenet_models/VGG16.caffemodel', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cfg', dest='cfg', help='Config file to overwrite the default configs',
                        default='SSH/configs/wider.yml', type=str)
    parser.add_argument('--iters', dest='iters', help='Number of iterations for training the network',
                        default=21000, type=int)

    return parser.parse_args()

if __name__ == '__main__':

    # Get command line arguments
    args = parser()

    # Combine external configs with SSH default configs
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg_print(cfg,test=False)

    # Set the GPU ids
    gpu_list = args.gpu_ids.split(',')
    gpus = [int(i) for i in gpu_list]

    # Set the random seed for numpy
    np.random.seed(cfg.RNG_SEED)

    # Prepare the training roidb
    imdb= get_imdb(args.db_name)
    roidb = get_training_roidb(imdb)

    # Train the model
    train_net(args.solver_proto, roidb, output_dir=get_output_dir(imdb.name),
              pretrained_model=args.pretrained,
              max_iter=args.iters, gpus=gpus)


