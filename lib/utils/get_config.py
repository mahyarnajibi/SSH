# --------------------------------------------------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# This file is a modified version from https://github.com/rbgirshick/py-faster-rcnn by Ross Girshick
# Modified by Mahyar Najibi
# --------------------------------------------------------------------------------------------------

from __future__ import print_function
import os
import os.path as osp
import yaml
import numpy as np
from easydict import EasyDict


# PARSE THE DEFAULT CONFIG
default_cfg_path = 'SSH/configs/default_config.yml'
assert osp.isfile(default_cfg_path), 'The default config is not found in {}!'.format(default_cfg_path)
with open(default_cfg_path, 'r') as cfg_file:
    cfg = EasyDict(yaml.load(cfg_file))

# CONVERT PIXEL_MEANS to numpy
cfg.PIXEL_MEANS = np.array(cfg.PIXEL_MEANS)

# SET ROOT DIRECTORY
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# FORM ADDRESS TO THE DATA DIRECTORY
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR,cfg.DATA_DIR)


def get_output_dir(imdb_name, net_name=None,output_dir='output'):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """

    outdir = osp.abspath(osp.join(cfg.ROOT_DIR, output_dir, cfg.EXP_DIR, imdb_name))
    if net_name is not None:
        outdir = osp.join(outdir, net_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def cfg_print(cfg, test=True):
    print('\x1b[32m\x1b[1m'+'#'*20+' Configuration '+'#'*20+'\x1b[0m')

    def cfg_print_recursive(cur_cfg, level=0):
        for k in cur_cfg:
            if test and k=='TRAIN':
                continue
            if not test and k=='TEST':
                continue
            if type(cur_cfg[k]) is EasyDict:
                print(' '*level*4+'\x1b[35m\x1b[1m' + k + '{' + '\x1b[0m')
                cfg_print_recursive(cur_cfg[k],level+1)
                print(' ' * level * 4 + '\x1b[35m\x1b[1m' + '}' + '\x1b[0m')
            else:
                print(' '*level*4+'\x1b[94m\x1b[1m'+k+':'+'\x1b[0m',end=' '*2)
                print(cur_cfg[k])

    cfg_print_recursive(cfg,0)
    print('\x1b[32m\x1b[1m' + '#' * (2*20 +len(' Configuration ')) + '\x1b[0m')


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = EasyDict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
