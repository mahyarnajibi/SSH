# --------------------------------------------------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# Utilities used in SSH test modules
# Written by Mahyar Najibi
# --------------------------------------------------------------------------------------------------
from utils.get_config import cfg
import numpy as np
import cv2
from utils.blob import im_list_to_blob


def _compute_scaling_factor(im_shape,target_size,max_size):
    """
    :param im_shape: The shape of the image
    :param target_size: The min side is resized to the target_size
    :param max_size: The max side is kept less than max_size
    :return: The scale factor
    """

    if cfg.TEST.ORIG_SIZE:
        return 1.0

    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def _get_image_blob(im,im_scales):
    """
    :param im: input image
    :param im_scales: a list of scale coefficients
    :return: A list of network blobs each containing a resized ver. of the image
    """
    # Subtract the mean
    im_copy = im.astype(np.float32, copy=True) - cfg.PIXEL_MEANS

    # Append all scales to form a blob
    blobs = []
    for scale in im_scales:
        if scale==1.0:
            blobs.append({'data':im_list_to_blob([im_copy])})
        else:
            blobs.append({'data':im_list_to_blob([cv2.resize(im_copy, None, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR)])})
    return blobs

