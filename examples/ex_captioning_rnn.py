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

# Debug
#from pudb import set_trace; set_trace()

def ex_caption_rnn(verbose=False, plot_figures=False):

    test_data = coco_utils.load_coco_data(max_train=5000)

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
