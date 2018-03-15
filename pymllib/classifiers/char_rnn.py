"""
CHAR_RNN
Adapation of Karpathy's character-level RNN

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pymllib.layers import rnn_layers

# debug
from pudb import set_trace; set_trace()

class CharRNN(object):
    """
    TODO: Docstring
    """
    def __init__(self, word_to_idx, **kwargs):
        self.verbose = kwargs.pop('verbose', False)
        self.clip = kwargs.pop('clip', True)
        self.word_to_idx = word_to_idx
        self.wordvec_dim = kwargs.pop('wordvec_dim', 100)
        self.hidden_dims = kwargs.pop('hidden_dims', 100)
        self.vocab_size = kwargs.pop('vocab_size', 100)
        self.weight_scale = kwargs.pop('weight_scale', 0.01)
        self.dtype = kwargs.pop('dtype', np.float32)

        # internal previous hidden state
        self.h_prev = None
        self.idx_to_word = {i: w for w, i, in self.word_to_idx.items()}

        self.params = {}
        # For now this implementation is fixed at one hidden layer
        # Weights
        self.params['Wxh'] = np.random.randn(self.hidden_dims, self.vocab_size)
        self.params['Whh'] = np.random.randn(self.hidden_dims, self.hidden_dims)
        self.params['Why'] = np.random.randn(self.vocab_size, self.hidden_dims)
        self.params['Wembed'] = np.random.randn(self.vocab_size, self.wordvec_dim)
        # Scale the params here
        for k, v in self.params.items():
            self.params[k] = v * self.weight_scale

        # Biases
        self.params['bh'] = np.zeros((self.hidden_dims, 1))
        self.params['by'] = np.zeros((self.vocab_size, 1))

        # Cast parameters to correct type
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, inputs, targets):

        loss = 0.0
        grads = {}
        Wxh = self.params['Wxh']
        Whh = self.params['Whh']
        Why = self.params['Why']
        bh = self.params['bh']
        by = self.params['by']
        Wembed = self.params['Wembed']

        #h0 = np.dot(inputs, Wxh) # TODO: This is not correct...
        if self.h_prev is None:
            self.h_prev = np.zeros((inputs.shape[0], self.hidden_dims))

        # 'embed' the input
        #x = np.zeros((self.vocab_size, 1))

        x, cache_embedding = rnn_layers.word_embedding_forward(inputs, Wembed)
        # ==== FORWARD PASS ==== #
        # where does h0 come from?
        h0 = self.h_prev
        h, cache_rnn = rnn_layers.rnn_forward(x, h0, Wxh, Whh, bh)
        scores, cache_scores = rnn_layers.temporal_affine_forward(h, Why, by)
        # find probs
        loss, dscores = rnn_layers.temporal_softmax_loss(
            scores, targets, verbose=self.verbose)

        grads = dict.fromkeys(self.params)
        # ==== BACKWARD PASS ==== #
        dh, dWhy, dby = rnn_layers.temporal_affine_backward(
            dscores, cache_scores)
        dx, dh0, dWxh, dWhh, dbh = rnn_layers.rnn_backward(dh, cache_rnn)

        # Reverse word embedding
        dWembed = rnn_layers.word_embedding_backward(dx, cache_embed)
        dWproj = np.dot(targets.T, dh)
        dbproj = np.sum(h0, axis=0)

        self.h_prev = h

        # Update grads
        grads['dx'] = dx
        grads['dWhy'] = dWhy
        grads['dWxh'] = dWxh
        grads['dWhh'] = dWhh
        grads['dby'] = dby
        grads['dbh'] = dbh
        grads['dh'] = dh
        grads['dWproj'] = dWproj

        if self.clip:       # Mitigate exploding gradients
            for k, p in self.grads.items():
                if k[0] == 'd':
                    np.clip(p, -5, 5, out=p)
                    if self.verbose:
                        print("Clipped parameter %s" % k)

        return loss, grads

    def sample(self, features, seed_idx, max_length=30):

        X = np.zeros((self.vocab_size, 1))
        X[seed_idx] = 1
        indexes = []
        # get params out of model
        Wxh = self.params['Wxh']
        Whh = self.params['Whh']
        Why = self.params['Why']
        bh = self.params['bh']
        by = self.params['by']
        h = features        # alias var
        for t in range(max_length):
            #h, cache_rnn = rnn_layers.rnn_forward() # TODO : use the existing forward pass
            h = np.tanh(np.dot(Wxh, X) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            # got new index, reset X
            X = np.zeros((self.vocab_size, 1))
            X[ix] = 1
            indexes.append(ix)

        return indexes
