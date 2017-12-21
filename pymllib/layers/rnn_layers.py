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
        - b  : Biases. Shape (H,)

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
        - cache : Values computed in forward pass.

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
    of V words assigning each to a vector of dimension D.

    Inputs:
        - x : Integer array of shape (N, T) giving indices of words. Each
        element of x must be in the range 0 <= idx < V
        - W : Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
        - out : Array of shape (N, T, D) giving word vectors for all input
        words.
        - cache : Values needed for backward pass
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



# ======== LSTM LAYERS ======== #

def lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass of a single timestep in an LSTM

    In input data has dimension D, the hidden state dimension H, and
    we operated on a minibatch size of N.

    Inputs:
        - x      : Input data. Shape (N, D)
        - prev_h : Previous hidden state. Shape (N, H)
        - prev_c : Previous cell state. Shape (N, H)
        - Wx     : Input-to-hidden weights. Shape (D, 4H)
        - Wh     : Hidden-to-hidden weights. Shape (H, 4H)
        - b      : Biases. Shape (4H,)

    Returns (as tuple):
        - next_h : Next hidden state. Shape (N, H)
        - next_c : Next cell state. Shape (N. H)
        - cache  : Tuple of values needed for backward pass

    """

    H = prev_h.shape[1]
    # compute intermediate vector
    a = np.dot(X, Wx) + np.dot(prev_h, Wh) + b

    # Select gates from range
    ai = a[:,   0:H]
    af = a[:,   H:2*H]
    ao = a[:, 2*H:3*H]
    ag = a[:, 3*H:4*H]
    # Compute gate activations
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)

    # compute next cell state
    next_c = f * prev_c + i * g
    # compute next hidden state
    next_h = o * np.tanh(next_c)
    # cache for backward pass
    cache = (i, f, o, g, a, ai, af, ao, ag,
             Wx, Wh, b, prev_h, prev_c, X,
             next_c, next_h)

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass of a single timestep in an LSTM

    Inputs:
        - dnext_h : Gradients of next hidden state. Shape (N, H)
        - dnext_c : Gradients of next cell state. Shape (N, H)
        - cache   : Variables from forward pass

    Returns (as tuple):
        - dx : Gradient of input data. Shape (N, D)
        - dprev_h  : Gradient of previous hidden state. Shape (N, H)
        - dprev_c  : Gradient of previous cell state. Shape (N, H)
        - dWx : Gradient of input-to-hidden weights. Shape (N, H)
        - dWh : Gradient of hidden-to-hidden weights. Shape (H, H)
        - db : Gradient of biases. Shape (4H,)

    """

    # Unpack cache (TODO : should this cache actually be a dict?)
    i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, X, next_c, next_h = cache
    # Backprop
    do = np.tanh(next_c) * dnext_h
    dnext_c += o * (1 - np.tanh(next_c)**2) * dnext_h

    # Forget gate
    df = prev_c * dnext_c
    dprev_c = f * dnext_c
    di = g * dnext_c
    dg = i * dnext_c

    # Next gate
    dag = (1 - np.tanh(ag)**2) * dg
    dao = sigmoid(ao) * (1 - sigmoid(ao)) * do
    daf = sigmoid(af) * (1 - sigmoid(df)) * df
    dai = sigmoid(ai) * (1 - sigmoid(ai)) * di

    # Backprop into activation
    da = np.hstack((dai, daf, dao, dag))
    # Backprop into input
    dx = np.dot(da, Wx.T)
    dWx = np.dot(X.T, da)
    dprev_h = np.dot(da, Wh.T)
    dWh = np.dot(prev_h.T, da)
    db = np.sum(da, axis=0)

    return dx, dprev_h, dprev_c, dWx, dWh, db



def lstm_forward(X, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We
    assume an input sequence composed of T vectors each of dimension
    D. Each step has a hidden size of H and operates on a minibatch of
    N sequences.

    The internal cell state is passed as input, but the initial cell state
    is set to zero. The cell state is not returned - we consider this as
    an internal variable of the LSTM and do not access it from the outside.

    Inputs:
        - x : Input data. Shape (N, T, D)
        - h0: Initial hidden state. Shape (N, H)
        - Wx : Weights for input-to-hidden connections. Shape (D, 4H)
        - Wh : Weights for hidden-to-hidden connections. Shape (H, 4H)
        - b  : Biases. Shape (4H,)

    Returns (as tuple):
        - h : Hidden states for all timesteps of all sequences. Shape (N, T, H)
        - cache : Values needed for backward pass

    """
    N, T, D = X.shape
    N, H = h0.shape

    # init internal state
    prev_h = h0
    prev_c = np.zeros_like(prev_h)
    h = np.zeros((T, N, H))
    X = X.transpose(1, 0, 2)        # Flip matrix
    cache = []

    # Run the loop over each element in the sequence
    for t in range(T):
        if t > 0:
            prev_h = h[t-1]
            prev_c = next_c
        h[t], next_c, cache_next = lstm_step_forward(
            X[t], prev_h, prev_c, Wx, Wh, b)
        cache.append(cache_next)

    h = h.transpose(1, 0, 2)        # Flip matrix back

    return h, cache



def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data

    Inputs:
        - dh : Gradients of upstream hidden states. Shape (N, T, D)
        - cache  : Values from forward pass

    Returns (as tuple):
        - dx : Gradient of input data. Shape (N, T, D)
        - dh0 : Gradient of initial hidden state. Shape (N, H)
        - dWx : Gradient of input-to-hidden weights. Shape (D, 4H)
        - dWh : Gradient of hidden-to-hidden weights. Shape (H, 4H)
        - db  : Gradient of biases. Shape (4H,)

    """

    # unpack cache
    i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, X, next_c, next_h = cache[0]
    N, T, H = dh.shape
    D = X.shape[-1]

    assert len(cache) == T      # TODO: raise exception?

    # Init gradients
    dx  = np.zeros((T, N, D))
    dh0 = np.zeros((N, H))
    db  = np.zeros((4 * H,))
    dWh = np.zeros((H, 4 * H))
    dWx = np.zeros((D, 4 * H))
    # Transpose dh
    dh = dh.transpose(1, 0, 2)
    dh_prev = np.zeros((N, H))
    dc_prev = np.zeros_like(dh_prev)

    # Compute backwards pass over sequence
    for t in reversed(range(T)):
        dh_current = dh[t] + dh_prev
        dc_current = dc_prev
        dx_t, dh_prev, dc_prev, dWx_t, dWh_t, db_t = lstm_step_backward(
            dh_current, dc_current, cache[t])
        dx[t] += dx_t
        dh0 = dh_prev
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    # Transpose matrix back
    dx = dx.transpose(1, 0, 2)

    return dx, dh0, dWx, dWh, db

