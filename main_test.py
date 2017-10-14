# -----------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Main module for evaluating the SSH on a given dataset
# Written by Mahyar Najibi
# -----------------------------------------------------

from SSH.test import test_net
import argparse
from datasets.factory import get_imdb
from utils.get_config import cfg, cfg_from_file, cfg_from_list,cfg_print
import caffe

def parser():
    parser = argparse.ArgumentParser('SSH Evaluate Module!',
                            description='You can change other configs by providing a YAML config file!')
    parser.add_argument('--db', dest='db_name', help='Path to the image',
                        default='wider_val', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='The GPU ide to be used',
                        default=0, type=int)
    parser.add_argument('--proto', dest='prototxt', help='SSH caffe test prototxt',
                        default='SSH/models/test_ssh.prototxt', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Output path for saving the figure',
                        default='output', type=str)
    parser.add_argument('--model', dest='model', help='SSH trained caffemodel',
                        default='data/SSH_models/SSH.caffemodel', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cfg', dest='cfg', help='Config file to overwrite the default configs',
                        default='SSH/configs/wider.yml', type=str)
    parser.add_argument('--vis', dest='visualize', help='visualize detections',
                        action='store_true')
    parser.add_argument('--net_name', dest='net_name',
                        help='The name of the experiment',
                        default='SSH',type=str)
    parser.add_argument('--no_cache', dest='no_cache', help='Do not cache detections',
                        action='store_true')
    parser.add_argument('--debug', dest='debug', help='Debug mode',
                        action='store_true')
    return parser.parse_args()


def main(args):
    # Combine the default config with
    # the external config file and the set command
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.DEBUG = args.debug
    cfg.GPU_ID = args.gpu_id
    cfg_print(cfg)

    # Loading the network
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)

    # Create the imdb
    imdb = get_imdb(args.db_name)

    # Set the network name
    net.name = args.net_name

    # Evaluate the network
    test_net(net, imdb, visualize=args.visualize, no_cache=args.no_cache, output_path=args.out_path)


if __name__ == '__main__':
    args = parser()
    main(args)
