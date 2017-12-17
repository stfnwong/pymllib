"""
RNN UTILS
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymllib.layers import rnn_layers
from pymllib.utils import coco_utils
from pymllib.utils import image_utils

import numpy as np
import matplotlib.pyplot as plt

def check_loss(N, T, V, p, verbose=False):
    x = 0.001 * np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = np.random.randn(N, T) <= p

    out = rnn_layers.temporal_softmax_loss(x, y, mask)[0]
    if verbose:
        print(out)

    return out


def test_time_sampling(data, model, batch_size=16):

    captions = {}
    for split in ['train', 'val']:
        minibatch = coco_utils.sample_coco_minibatch(data,
                        split=split,
                        batch_size=batch_size)
        gt_captions, features, urls = minibatch
        # Ground-truth captions
        gt_captions = coco_utils.decode_captions(gt_captions, data['idx_to_word'])
        # Sample captions
        sample_captions = model.sample(features)
        sample_captions = sample_captions.astype(np.int32)
        sample_captions = coco_utils.decode_captions(sample_captions, data['idx_to_word'])

        captions['%s-sample' % str(split)] = sample_captions
        captions['%s-gt' % str(split)] = gt_captions

    return captions

def plot_test_time_samples(gt_captions, sample_captions, urls, split):

    for gt, samp, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_utils.image_from_url(url))
        plt.title('Split : %s\nSample : %s\nGT: %s' % (split, samp, gt))
        plt.axis('off')
        plt.show()
