"""
CHAR_RNN
Adapation of Karpathy's character-level RNN

Stefan Wong 2017
"""

import numpy as np
from pymllib.layers import rnn_layers

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union


class CharRNN:
    """
    TODO: Docstring
    """
    def __init__(self, word_to_idx:Dict[str, int], **kwargs) -> None:
        self.verbose      = kwargs.pop('verbose', False)
        self.clip         = kwargs.pop('clip', True)
        self.word_to_idx  = word_to_idx
        self.hidden_dims  = kwargs.pop('hidden_dims', 100)
        self.vocab_size   = kwargs.pop('vocab_size', 100)
        self.weight_scale = kwargs.pop('weight_scale', 0.01)
        self.dtype        = kwargs.pop('dtype', np.float32)

        # internal previous hidden state
        self.h_prev = None

        self.params:Dict[str, Any]= {}
        # For now this implementation is fixed at one hidden layer
        # Weights
        self.params['Wxh'] = np.random.randn(self.hidden_dims, self.vocab_size)
        self.params['Whh'] = np.random.randn(self.hidden_dims, self.hidden_dims)
        self.params['Why'] = np.random.randn(self.vocab_size, self.hidden_dims)
        # Scale the params here
        for k, v in self.params.items():
            self.params[k] = v * self.weight_scale

        # Biases
        self.params['bh'] = np.zeros((self.hidden_dims, 1))
        self.params['by'] = np.zeros((self.vocab_size, 1))

        # Cast parameters to correct type
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self,
             inputs:np.ndarray,
             targets:np.ndarray,
             hprev:Union[None, np.ndarray]=None) -> Tuple[Any, Any]:

        if hprev is not None:
            self.hprev = np.copy(hprev)

        loss = 0.0
        grads = {}
        Wxh = self.params['Wxh']
        Whh = self.params['Whh']
        Why = self.params['Why']
        bh = self.params['bh']
        by = self.params['by']

        h0 = np.dot(inputs, Wxh) # TODO: This is not correct...

        h, cache_rnn = rnn_layers.rnn_forward(x, h0, Wxh, Whh, bh)
        scores, cache_scores = rnn_layers.temporal_affine_forward(h, Why, by)
        loss, dscores = rnn_layers.temporal_softmax_loss(
            scores, targets, verbose=self.verbose)

        # Backward pass
        dx, dh0, dWxh, dWhh, dbh = rnn_layers.rnn_backward(dscores, cache_rnn)

        # ==== FORWARD PASS ==== #
        for t in range(len(inputs)):
            xs = np.zeros((self.vocab_size, 1))
            xs[inputs[t]] = 1
            hs = np.tanh(np.dot(Wxh, xs) + np.dot(Whh, self.h_prev) + bh)
            # un-normalized log probs for next char
            ys = np.dot(Why, hs) + by
            ps = np.exp(ys) / np.sum(np.exp(ys))    # probs for next chars
            loss += np.log(ps[targets[t], 0])

        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dWhy = np.zeros_like(Why)
        dbh = np.zeros_like(bh)
        dby = np.zeros_like(by)
        dhnext = np.zeros_like(hs)

        # ==== BACKWARD PASS ==== #
        for l in reversed(range(len(inputs))):
            dy = np.copy(ps)
            dy[targets[l]] -= 1
            dWhy += np.dot(dy, hs.T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs * hs) * dh      # backprop through tanh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs.T)
            dWhh += np.dot(dhraw, hs.T)
            dhnext = np.dot(Whh.T, dhraw)


        # Update grads
        grads['dy'] = dy
        grads['dWhy'] = dWhy
        grads['dWxh'] = dWxh
        grads['dWhh'] = dWhh
        grads['dby'] = dby
        grads['dh'] = dh

        if self.clip:       # Mitigate exploding gradients
            for k, p in grads.items():
                if k[0] == 'd':
                    np.clip(p, -5, 5, out=p)
                    if self.verbose:
                        print("Clipped parameter %s" % k)

        return loss, grads

    def sample(self, h, seed_idx):
        pass        # shut linter up
