"""
EX_CHAR_RNN
Example of trainnig the character modelling RNN.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pymllib.classifiers import char_rnn
from pymllib.solver import solver       # TODO : Is the standard solver useful for this?
#from pymllib.solver import captioning_solver as cap_solver


def load_text_data(filename):
    with open(filename, 'r') as fp:
        data = fp.read()

    return data


def run_char_rnn(verbose=True):

    # Get some data
    data = load_text_data('datasets/shakespear.txt')
    chars = list(set(data))
    data_size = len(data)
    vocab_size = len(chars)
    hidden_dims = 100
    seq_length = 25     # number of steps to unroll RNN for

    print("Data has %d characters, %d unique" % (data_size, vocab_size))
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    rnn = char_rnn.CharRNN(verbose=verbose,
                           hidden_dims=hidden_dims,
                           vocab_size=vocab_size,
                           word_to_idx=char_to_idx)


    # Prepare inputs
    n = 0
    p = 0
    # prepare inputs. We sweep from left to right in steps of seq_length
    if p + seq_length + 1 > len(data) or n == 0:
        hprev = np.zeros((hidden_dims, 1))      # reset RNN memory
        p = 0

    inputs = [char_to_idx[ch] for ch in data[p: p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1: p+seq_length+1]]

    # TODO : also need to sample from the model from time to time
    if n % 100 == 0:
        sample_ix = rnn.sample(hprev, inputs[0])

    num_iters = 50
    for t in range(num_iters):
        loss, grads = rnn.loss(inputs, targets)



    # Cant use regular solver here
    solv = solver.CaptioningSolver(rnn, data)
    solv.train()


if __name__ == '__main__':
    run_char_rnn()
