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

# Debug
from pudb import set_trace; set_trace()

def load_text_data(filename):
    with open(filename, 'r') as fp:
        data = fp.read()

    return data


def run_char_rnn(verbose=True):

    # Get some data
    sep_str = '================'
    zdata = load_text_data('datasets/shakespear.txt')
    chars = list(set(zdata))
    data_size = len(zdata)
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
    if p + seq_length + 1 > len(zdata) or n == 0:
        hprev = np.zeros((hidden_dims, 1))      # reset RNN memory
        p = 0

    # Change to numpy array
    # shape should be N, T, D (minibatch, seq_length, vocab)
    #inputs = np.zeros((1, seq_length, vocab_size))
    #targets = np.zeros((1, seq_length, vocab_size))

    #for k, ch in enumerate(zdata[p: p+seq_length]):
    #    inputs[0, k, 0] = char_to_idx[ch]

    #for k, ch in enumerate(zdata[p+1: p+seq_length+1]):
    #    targets[0, k, 0] = char_to_idx[ch]

    inputs = np.zeros((seq_length, vocab_size), dtype=np.int32)
    targets = np.zeros((seq_length, vocab_size), dtype=np.int32)

    for k, ch in enumerate(zdata[p: p+seq_length]):
        inputs[k, 0] = char_to_idx[ch]

    for k, ch in enumerate(zdata[p+1: p+seq_length+1]):
        targets[k, 0] = char_to_idx[ch]


    #inputs = [char_to_idx[ch] for ch in zdata[p: p+seq_length]]
    #targets = [char_to_idx[ch] for ch in zdata[p+1: p+seq_length+1]]

    # TODO : also need to sample from the model from time to time
    if n % 100 == 0 and n > 0:
        sample_ix = rnn.sample(hprev, inputs[0])
        sample_txt = []
        for ix in sample_ix:
            sample_txt.append(idx_to_char[ix])
        print('%s\n %s \n%s\n' % (sep_str, ''.join(sample_txt), sep_str))

    num_iters = 50
    for t in range(num_iters):
        loss, grads = rnn.loss(inputs.T, targets)



    # Cant use regular solver here
    solv = solver.CaptioningSolver(rnn, zdata)
    solv.train()


if __name__ == '__main__':
    run_char_rnn()
