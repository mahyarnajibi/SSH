# --------------------------------------------------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# This file is a modified version from https://github.com/rbgirshick/py-faster-rcnn
# Modified by Mahyar Najibi
# --------------------------------------------------------------------------------------------------

import caffe
import numpy as np
import numpy.random as npr
import yaml
from utils.cython_bbox import bbox_overlaps

from utils.get_config import cfg
from utils.bbox_transform import bbox_transform
from SSH.layers.generate_anchors import generate_anchors


class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        # Determine if hard negative mining should be performed
        # based on the number of bottom blobs
        self._hard_mining = False
        if len(bottom) == 5:
            self._hard_mining = True
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchor_ratios = layer_params.get('ratios',(0.5, 1, 2))
        base_size = layer_params.get('base_size', 16)
        self._anchors = generate_anchors(scales=np.array(anchor_scales), base_size=base_size,
                                         ratios=np.array(self._anchor_ratios))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']
        self._positive_overlap = layer_params.get('positive_overlap',cfg.TRAIN.ANCHOR_POSITIVE_OVERLAP)

        # allow boxes to sit over the edge
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        # keep only inside anchors
        if inds_inside.shape[0]==0:
            # If no anchors inside use whatever anchors we have
            inds_inside = np.arange(0,all_anchors.shape[0])

        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.TRAIN.ANCHOR_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        if cfg.TRAIN.FORCE_FG_FOR_EACH_GT:
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]

            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
            labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self._positive_overlap] = 1

        # Subsample positives
        num_fg = int(cfg.TRAIN.ANCHOR_FG_FRACTION * cfg.TRAIN.ANCHORS_PER_BATCH)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            if self._hard_mining and cfg.TRAIN.HARD_POSITIVE_MINING:
                ohem_scores = bottom[4].data[:, self._num_anchors:, :, :]
                ohem_scores = ohem_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                ohem_scores = ohem_scores[inds_inside]
                pos_ohem_scores = 1 - ohem_scores[fg_inds]
                order_pos_ohem_scores = pos_ohem_scores.ravel().argsort()[::-1]
                ohem_sampled_fgs = fg_inds[order_pos_ohem_scores[:num_fg]]
                labels[fg_inds] = -1
                labels[ohem_sampled_fgs] = 1
            else:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

        # Subsample negatives
        n_fg = np.sum(labels == 1)
        num_bg = cfg.TRAIN.ANCHORS_PER_BATCH - n_fg
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            if not self._hard_mining:
                # randomly sub-sample negatives
                disable_inds = npr.choice(
                    bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels[disable_inds] = -1
            else:
                # sort ohem scores
                ohem_scores = bottom[4].data[:, self._num_anchors:, :, :]
                ohem_scores = ohem_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                ohem_scores = ohem_scores[inds_inside]
                neg_ohem_scores = ohem_scores[bg_inds]
                order_neg_ohem_scores = neg_ohem_scores.ravel().argsort()[::-1]
                ohem_sampled_bgs = bg_inds[order_neg_ohem_scores[:num_bg]]
                labels[bg_inds] = -1
                labels[ohem_sampled_bgs] = 0

        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
