# --------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Written by Mahyar Najibi
# --------------------------------------------------------

import sys
# Add caffe and lib to the paths
if not 'caffe-ssh/python' in sys.path:
    sys.path.insert(0,'caffe-ssh/python')
if not 'lib' in sys.path:
    sys.path.insert(0,'lib')
from utils.get_config import cfg

if not cfg.DEBUG:
    import os
    # Suppress Caffe (it does not affect training, only test and demo)
    os.environ['GLOG_minloglevel']='3'
