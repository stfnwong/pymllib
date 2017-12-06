"""
RNN_LAYERS
Layer functions for recurrent neural networks
"""

import numpy as np

# Debug
#from pudb import set_trace; set_trace()

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Forward pass for a single timestep of a vanilla RNN that uses
    a tanh activation function. Input data has dimension D, hidden
    state has dimension H, each minibatch has size N.

    Inputs:
        - X      : Input data for this timestep. Shape (N, D)
        - prev_h : Hidden state from previous timestep. Shape (N, H)
        - Wx     : Weight matrix for input-to-hidden connections. Shape (D, H)
        - Wh     : Weight matrix for hidden-to-hidden connections. Shape (H, H)
        - b      : Biases. Shape (H,)

    Returns tuple of:
        - next_h : Next hidden state of shape (N, H)
        - cache  : Tuple of values needed for backward pass
    """

    forward = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h = np.tanh(forward)
    cache = (x, Wx, prev_h, Wh, forward)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
        - dnext_h : Gradient of loss with respect to next
        hidden state.
        - cache : Tuple of (x, Ws, prev_h, Wh, forward)

    Returns a tuple of:
        - dx      : Gradients of input data. Shape (N, D)
        - dprev_h : Gradients of previous hidden state. Shape (N, H)
        - dWx     : Gradients of input-to-hidden weights. Shape (N, H)
        - dWh     : Gradients of hidden-to-hidden weights. Shape (H, H)
        - db      : Gradients of bias vector. Shape (H,)
    """

    x, Wx, prev_h, Wh, forward = cache
    # Backprop the forward pass
    dforward = (1 - np.tanh(forward)**2) * dnext_h

    dx = np.dot(dforward, Wx.T)
    dWx = np.dot(x.T, dforward)
    dprev_h = np.dot(dforward, Wh.T)
    dWh = np.dot(prev_h.T, dforward)
    db = np.sum(dforward, axis=0)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an
    input sequence composed of T vectors, each of dimension D. The RNN uses
    a hidden size of H, and we operated on a minibatch containing N sequences.
    After running the RNN forward we return the hidden state for all
    timesteps.

    Inputs:
        - x  : Input data for the entire timeseries. Shape (N, T, D)
        - h0 : Initial hidden state. Shape (N, H)
        - Wx : Weight matrix for input-to-hidden connections. Shape (D, H)
        - Wh : Weight matrix for hidden-to-hidden connections. Shape (H, H)
        - b  : Biases. Shape (H,)c

    Returns:
        - h : Hidden states for entire timeseries. Shape (N, T, H)
        - cache : Values required for backward pass

    """
    N, T, D = x.shape
    H = h0.shape[1]
    cache = []

    x = x.transpose(1, 0, 2)
    h = np.zeros((T, N, H))

    h[-1] = h0
    for t in range(T):
        if t == 0:
            h_prev = h0
        else:
            h_prev = h[t-1]
        h[t], cache_next = rnn_step_forward(x[t], h_prev, Wx, Wh, b)
        cache.append(cache_next)

    # Since x was transposed, transpose the hidden vector back
    h = h.transpose(1, 0, 2)

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
        - dh : Upstream gradients. Shape (N, T, H)
        - cache : Values computed in forward pass. Eac

    Returns:
        - dx: Gradient of inputs. Shape (N, T, D)
        - dh0 : Gradient of initial hidden state. Shape (N, H)
        - dWx : Gradient of input-to-hidden weights. Shape (D, H)
        - dWh : Gradient of hidden-to-hidden weights. Shape (H, H)
        - db  : Gradient of biases. Shape (H,)
    """

    N, T, H = dh.shape
    D = cache[0][0].shape[1]

    # Init gradients
    dx = np.zeros((T, N, D))
    dh0 = np.zeros((N, H))
    db = np.zeros((H))
    dWh = np.zeros((H, H))
    dWx = np.zeros((D, H))

    # Transpose dh
    dh = dh.transpose(1, 0, 2)
    dh_prev = np.zeros((N, H))

    for t in reversed(range(T)):
        dh_current = dh[t] + dh_prev
        dx_t, dh_prev, dWx_t, dWh_t, db_t = rnn_step_backward(
            dh_current, cache[t])
        dx[t] += dx_t
        dh0 = dh_prev
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    # Restore the gradient matrix
    dx = dx.transpose(1, 0, 2)

    return dx, dh0, dWx, dWh, db

def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set
    of D-dimensional vectors arranged into a minibatch of N
    timeseries, each of length T. We use an affine function to
    transform each of those vectors into a new vector of
    dimension M.

    Inputs:
        - x: Input data of shape (N, T, D)
        - w: Weights of shape (D, M)
        - b: Biases of shape (M,)

    Returns a tuple of:
        - out : Output data of shape (N, T, M)
        - cache : Values needed for the backward pass. cache[n]
        contains values computed in the forward pass for timestep n.
    """

    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = (x, w, b, out)

    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for a temporal affine layer.

    Inputs:
        - dout : Upstream gradients of shape (N, T, M)
        - cache : Values from forward pass

    Returns:
        - dx : Gradient of input, shape (N, T, D)
        - dw : Gradient of weights, shape (D, M)
        - db : Gradient of biases,  shape (M,)

    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that
    we are making predictions over a vocabulary of size V for each
    timestep of a timeseries of length T over a minibatch of size N.
    The input X gives scores for all vocabulary elements at all timesteps,
    and y gives the indicies of the ground-truth element at each timestep
    We use a cross-entropy loss at each timestep, summing the loss over
    all timesteps and averaging across the minibatch.

    Additionally, we may want to ignore the model output at some timesteps,
    since sequences of different length may have been combined into a
    minibatch and padded with <NULL> tokens. The mask argument indicates
    which elements should contribute to the loss.

    Inputs:
        - x: Input scores. Shape (N, T, V)
        - y: Ground-truth indicies. Shape (N, T) where each element is in
        the range 0 <= y[i, t] < V
        - mask: Boolean array of shape (N, T) where mask[i, t] tells whether
        or not the scores at x[i, t] should contribute to the loss.

    Returns:
        - loss: Scalar loss
        - dx : Gradient of loss with respect to x

    """
    N, T, V = x.shape
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N

    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat : %s" % str(dx_flat.shape))

    dx = dx_flat.reshape(N, T, V)

    return loss, dx


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of
    size N where each sequence has length T. We assume a vocabulary
    of V words
    """

    out = W[x, :]
    cache = x, W

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. Because the words are integers we
    cannot backpropogate into them, so we only return gradient for the
    word embedding matrix
    """

    print('word_embedding_backward : %d' % len(cache))
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function
    """

    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]

    return top / (1 + z)
