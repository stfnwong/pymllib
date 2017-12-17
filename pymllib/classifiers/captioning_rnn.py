"""
CAPTIONING_RNN

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pymllib.layers import layers
from pymllib.layers import rnn_layers

# Debug
#from pudb import set_trace; set_trace()

class CaptioningRNN(object):
    def __init__(self, word_to_idx, **kwargs):
        """
        CAPTIONING RNN

        Inputs:
            - word_to_idx : A dictionary giving the vocabulary. It contains
            V entries and maps each string to a unique integer in the
            range [0, V)
            - input_dim : Dimension D of input feature vectors
        """
        self.verbose = kwargs.pop('verbose', False)

        # Keyword params
        self.word_to_idx = word_to_idx
        self.input_dim = kwargs.pop('input_dim', 512)
        self.wordvec_dim = kwargs.pop('wordvec_dim', 128)
        self.hidden_dim = kwargs.pop('hidden_dim', 128)
        self.cell_type = kwargs.pop('cell_type', 'rnn')
        self.dtype = kwargs.pop('dtype', np.float32)

        if self.cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell type %s' % self.cell_type)

        # Internal params
        self.idx_to_word = {i: w for w, i, in self.word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)
        # Special tokens
        self._null = self.word_to_idx['<NULL>']
        self._start = self.word_to_idx.get('<START>', None)
        self._end = self.word_to_idx.get('<END>', None)
        # Init word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, self.wordvec_dim)
        self.params['W_embed'] /= 100
        # Init CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(self.input_dim, self.hidden_dim)
        self.params['W_proj'] /= np.sqrt(self.input_dim)
        self.params['b_proj'] = np.zeros(self.hidden_dim)
        # Init parameters for RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[self.cell_type]
        self.params['Wx'] = np.random.randn(self.wordvec_dim, dim_mul * self.hidden_dim)
        self.params['Wx'] /= np.sqrt(self.wordvec_dim)
        self.params['Wh'] = np.random.randn(self.hidden_dim, dim_mul * self.hidden_dim)
        self.params['Wh'] /= np.sqrt(self.hidden_dim)
        self.params['b'] = np.zeros(dim_mul * self.hidden_dim)
        # Init output to vocab weights
        self.params['W_vocab'] = np.random.randn(self.hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(self.hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameter data types
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training time loss.

        Inputs:
            - features: Input image features. Shape (N, D)
            - captions: Integer array of ground-truth captions. Shape (N, T).
            Each element is in the range 0 <= y[i, t] < V

        """

        # Cut captions into two peices.
        # captions_in contains all but the last word and will be input to
        # the RNN.
        # captions_out contains all but the first word. This is what the RNN
        # is expected to generate. These are offset by one relative to each
        # other because the RNN should produce word (t+1) after receiving word
        # t. The first element of captions_in will be the START token, and the
        # first element of captions_out will be the first word
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self._null)
        # Weights and bias for the arrine transform from image features to
        # initial hidden state
        W_proj = self.params['W_proj']
        b_proj = self.params['b_proj']
        # Word embedding matrix
        W_embed = self.params['W_embed']
        # Input to hidden, hidden-to-hidden, and biases for the RNN
        Wx = self.params['Wx']
        Wh = self.params['Wh']
        b = self.params['b']
        # Weight and bias for hidden-to-vocab transformation
        W_vocab = self.params['W_vocab']
        b_vocab = self.params['b_vocab']

        loss = 0.0
        grads = {}

        # ===============================
        # FORWARD PASS
        # ===============================
        h0 = np.dot(features, W_proj) + b_proj
        #print('type W_embed : %s' % type(W_embed))
        x, cache_embedding = rnn_layers.word_embedding_forward(captions_in, W_embed)

        if self.cell_type == 'rnn':
            h, cache_rnn = rnn_layers.rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            h, cache_rnn = rnn_layers.lstm_forward(x, h0, Wx, Wh, b)
        else:
            raise ValueError('cell_type %s not implemented' % self.cell_type)

        scores, cache_scores = rnn_layers.temporal_affine_forward(h, W_vocab, b_vocab)
        # Compute scores
        loss, dscores = rnn_layers.temporal_softmax_loss(scores,
                                                         captions_out, mask, verbose=self.verbose)
        # ===============================
        # BACKWARD PASS
        # ===============================
        grads = dict.fromkeys(self.params)

        dh, dW_vocab, db_vocab = rnn_layers.temporal_affine_backward(
            dscores, cache_scores)
        if self.cell_type == 'rnn':
            dx, dh0, dWx, dWh, db = rnn_layers.rnn_backward(dh, cache_rnn)
        elif self.cell_type == 'lstm':
            dx, dh0, dWx, dWh, db = rnn_layers.lstm_backward(dh, cache_rnn)
        else:
            raise ValueError('cell_type %s not implemented' % self.cell_type)

        dW_embed = rnn_layers.word_embedding_backward(dx, cache_embedding)
        dW_proj = np.dot(features.T, dh0)
        db_proj = np.sum(dh0, axis=0)

        # add everything to dict
        grads['W_proj'] = dW_proj
        grads['b_proj'] = db_proj
        grads['W_embed'] = dW_embed
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass the current word and
        the previous hidden state to the RNN to get the next hidden state, use
        the hidden state to get scores from all words in the vocabulary, and
        then choose the word with the highest score as the next word. The inital
        hidden state if computed by applying an affine transform to the input
        image features, and the initial word is the <START> token.

        For LTSMs, we must also keep track of the cell state. In that case the
        initial cell state should be zero.

        Inputs:
            - features: An array of input images features of shape (N, D)
            - max_length : maximum length T of generated captions

        Returns:
            - captions : Array of shape (N, max_length) giving sampled
            captions, where each element is an integer in the range [0, V).
            The first element of captions should be the first sampled word,
            not the <START> token.
        """

        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=self.dtype)

        # Unpack parameters
        W_proj = self.params['W_proj']
        b_proj = self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx = self.params['Wx']
        Wh = self.params['Wh']
        b = self.params['b']
        W_vocab = self.params['W_vocab']
        b_vocab = self.params['b_vocab']

        # Get first hidden state
        h0 = np.dot(features, W_proj) + b_proj
        captions[:, 0] = self._start
        # init previous state
        prev_h = h0
        prev_c = np.zeros_like(h0)
        # current word
        cur_word = self._start * np.ones((N, 1))

        # Iterate over the sequence
        for t in range(max_length):
            cur_word = cur_word.astype(np.int32) # Convert type
            word_embed, _ = rnn_layers.word_embedding_forward(cur_word, W_embed)
            if self.cell_type == 'rnn':
                h, _ = rnn_layers.rnn_step_forward(
                    np.squeeze(word_embed), prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                # Run one step of LSTM
                h, c, _ = rnn_layers.lstm_step_forward(np.squeeze(word_embed),
                                                       prev_h, prev_c, Wx, Wh, b)
            else:
                raise ValueError('cell_type %s not implemented' % self.cell_type)

            # Compute score distribution over dictionary
            scores, _ = rnn_layers.temporal_affine_forward(
                h[:, np.newaxis, :], W_vocab, b_vocab)
            # Squeeze out un-needed dimensions, find best word index
            idx_best = np.squeeze(np.argmax(scores, axis=2))
            captions[:, t] = idx_best
            # Update the hidden state, cell state (lstm only) and current word
            prev_h = h
            if self.cell_type == 'lstm':
                prev_c = c
            cur_word = captions[:, t]

        return captions
