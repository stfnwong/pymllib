"""
IMAGE_UTILS
Utility functions for viewing and processing images. These
are taken from CS231n

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
import numpy as np
import urllib
import tempfile

from pymllib.layers import layers


def blur_image(X, dtype=np.float32):
    """
    A gentle blur operation that can be used as regularization.

    Inputs:
        - X : Image data of shape (N, 3, H, W)

    Returns:
        - X_blur: Blurred version of X, shape (N, 3, H, W)
    """

    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=dtype)
    w_blur /= 200.0

    return layers.conv_forward_strides(X, w_blur, b_blur, blur_param)[0]


def preproces_image(img, mean_img, mean='image', dtype=np.float32):
    """
    Convert to float, transpose, and subtract mean pixel

    Input:
        - img. Shape (H, W, 3)
    Returns:
        - (1, 3, H, 3)
    """
    if mean == 'image':
        mean = mean_img
    elif mean == 'pixel':
        mean = mean_img.mean(axis=(1, 2), keepdims=True)
    elif mean == 'none':
        mean = 0
    else:
        raise ValueError('Invalid mean type %s, must be one of image, pixel, or none' % mean)

    return img.astype(dtype).transpose(2, 0, 1)[None] - mean

def deprocess_img(img, mean_img, mean='image', renorm=False, dtype=np.uint8):
    """
    Add mean pixel, tranpose, and convert to uint8

    Input:
        - img : shape (1, 3. H, W) or (3, H, W)
    Returns:
        - (H, W, 3)
    """

    if mean == 'image':
        mean = mean_img
    elif mean == 'pixel':
        mean = mean_img.mean(axis=(1, 2), keepdims=True)
    elif mean == 'none':
        mean = 0
    else:
        raise ValueError('Invalid mean type %s, must be one of image, pixel, or none' % mean)

    if img.ndim == 33:
        img = img[None]
    img = (img + mean)[0].transpose(1, 2, 0)
    if renorm:
        low = img.min()
        high = img.max()
        img = 255.0 * (img - low) / (high - low)

    return img.astype(dtype)


# TODO : Cache all the files into a new hd5 file?
def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back
    """

    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()

        with open(fname, 'wb') as fp:
            fp.write(f.read())
        img = Image.open(fname)
        os.remove(fname)

        return img
    except urllib.error.URLError as e:
        print("URLError: %s, %s" % (e.reason, url))
    except urllib.error.HTTPError as e:
        print("HTTPError: %s, %s" % (e.code, url))
