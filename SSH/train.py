# --------------------------------------------------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Main training module
# This file is a modified version from https://github.com/rbgirshick/py-faster-rcnn
# Modified by Mahyar Najibi
# --------------------------------------------------------------------------------------------------
import os
import caffe
import google.protobuf.text_format as text_format
from multiprocessing import Process
from utils.get_config import cfg
import numpy as np
from caffe.proto import caffe_pb2
import google.protobuf as pb2


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """
    def __init__(self, solver_prototxt, roidb, output_dir, gpu_id,
                 pretrained_model=None):
        """
        :param solver_prototxt: Solver prototxt
        :param roidb: The training roidb
        :param output_dir: Output directory for saving the models
        :param gpu_id: GPU id for the current process
        :param pretrained_model: The pre-trained model
        """
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.solver = caffe.SGDSolver(solver_prototxt)

        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            try:
                pb2.text_format.Merge(f.read(), self.solver_param)
            except:
                text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb,gpu_id)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print('Wrote snapshot to: {:s}'.format(filename))
        return filename


    def get_solver(self):
        return self.solver


def worker(rank, uid, gpus, solver_prototxt, roidb, pretrained_model, max_iter, output_dir):
    """
    Training worker
    :param rank: The process rank
    :param uid: The caffe NCCL uid
    :param solver_proto: Solver prototxt
    :param roidb: Training roidb
    :param pretrained_model: Pretrained model
    :param gpus: GPUs to be used for training
    :param max_iter: Maximum number of training iterations
    :param output_dir: Output directory used for saving models
    :return:
    """

    # Setup caffe
    caffe.set_device(gpus[rank])
    caffe.set_mode_gpu()
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)
    cfg.GPU_ID = gpus[rank]

    # Setup Solver
    solverW = SolverWrapper(solver_prototxt=solver_prototxt, roidb=roidb, output_dir=output_dir,gpu_id=rank,pretrained_model=pretrained_model)
    solver = solverW.get_solver()
    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    # Train the model for the specified number of iterations
    while solver.iter < max_iter:
        solver.step(1)
        if (solver.iter%cfg.TRAIN.SNAPSHOT == 0 or solver.iter == max_iter-1) and rank == 0:
            # Snapshot only in the main process
            solverW.snapshot()



def get_training_roidb(imdb):
    """
    Get the training roidb given an imdb
    :param imdb: The training imdb
    :return: The training roidb
    """
    def filter_roidb(roidb):
        """
        Filtering samples without positive and negative training anchors
        :param roidb: the training roidb
        :return: the filtered roidb
        """
        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.TRAIN.ANCHOR_POSITIVE_OVERLAP)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (overlaps >= cfg.TRAIN.BG_THRESH_LOW))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid

        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        return filtered_roidb

    # Augment imdb with flipped images
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    # Add required information to imdb
    imdb.prepare_roidb()
    # Filter the roidb
    final_roidb = filter_roidb(imdb.roidb)
    print('done')
    return final_roidb


def train_net(solver_prototxt, roidb, output_dir, pretrained_model, max_iter, gpus):
    """
    Training the network with multiple gpu
    :param solver_prototxt: the network prototxt
    :param roidb: the training roidb
    :param output_dir: the output directory to be used for saving the models
    :param pretrained_model: the pre-trained model for fine-tuning
    :param max_iter: maximum number of iterations for solver
    :param gpus: the GPU ids to be used for solving
    :return:
    """

    # Initiate Caffe NCCL
    uid = caffe.NCCL.new_uid()
    caffe.init_log(0,True)
    caffe.log('Using devices %s' % str(gpus))
    # Create a process per GPU
    procs = []
    for rank in range(len(gpus)):
        p = Process(target=worker,
                    args=(rank, uid, gpus, solver_prototxt, roidb, pretrained_model, max_iter, output_dir))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join() 
    print('done solving!')
