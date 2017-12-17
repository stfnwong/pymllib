"""
EX_CAPTIONING_RNN
Example with captioning RNN

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
# Modules
from pymllib.classifiers import captioning_rnn
from pymllib.solver import captioning_solver
from pymllib.utils import coco_utils
from pymllib.utils import image_utils

# Debug
#from pudb import set_trace; set_trace()

def test_time_sampling(data, model, batch_size=16, plot_figures=False):

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

        if plot_figures:
            for gt, samp, url in zip(gt_captions, sample_captions, urls):
                plt.imshow(image_utils.image_from_url(url))
                plt.title('Split : %s\nSample : %s\nGT: %s' % (split, samp, gt))
                plt.axis('off')
                plt.show()

def ex_caption_rnn(verbose=False, plot_figures=False):

    test_data = coco_utils.load_coco_data(max_train=50000)

    small_rnn_model = captioning_rnn.CaptioningRNN(
        cell_type='rnn',
        word_to_idx=test_data['word_to_idx'],
        input_dim=test_data['train_features'].shape[1],
        hidden_dim=512,
        wordvec_dim=256
    )

    solv = captioning_solver.CaptioningSolver(
        small_rnn_model,
        test_data,
        update_rule='adam',
        num_epochs=50,
        batch_size=25,
        optim_config={'learning_rate': 5e-3},
        lr_decay=0.95,
        verbose=verbose,
        print_every=100
    )

    solv.train()
    test_time_sampling(test_data,
                       small_rnn_model,
                       plot_figures=plot_figures)
    # Plot the loss...
    if plot_figures:
        plt.plot(solv.loss_history, 'o')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Captioning RNN Loss')
        plt.show()


if __name__ == '__main__':
    global_verbose = True
    global_plot = True
    ex_caption_rnn(verbose=global_verbose,
                   plot_figures=global_plot)
