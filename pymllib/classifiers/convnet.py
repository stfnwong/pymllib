"""
CONVNET
Some basic convolutional networks

"""
import numpy as np
from pymllib.layers import layers
from pymllib.layers import conv_layers
from pymllib.utils import layer_utils

from typing import Dict

# Debug
#from pudb import set_trace; set_trace()

# Some helper functions... there are really only for debugging use and
# should be removed later in this branch
def print_h_sizes(blocks):
    for k, v, in blocks.items():
        if k[:1] == 'h':
            print("%s : %s " % (str(k), str(v.shape)))

def print_layers(params, layer_type='W'):
    for k, v in params.items():
        if k[:1] == layer_type:
            print("%s : %s " % (str(k), str(v.shape)))


class ConvNetLayer:
    """
    An L-layer convolutional network with the following architecture

    [conv-relu-pool2x2] X L - [affine - relu] x M - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels
    """

    def __init__(self, **kwargs) -> None:
        """
        Init a new network
        """

        # Get kwargs
        self.verbose = kwargs.pop('verbose', False)
        self.use_batchnorm = kwargs.pop('use_batchnorm', False)
        #self.use_xavier = kwargs.pop('use_xavier', False)
        self.weight_init = kwargs.pop('weight_init', 'gauss')
        # TODO : Dropout?
        self.reg = kwargs.pop('reg', 0.0)
        self.weight_scale = kwargs.pop('weight_scale', 1e-3)
        self.dtype = kwargs.pop('dtype', np.float32)
        self.filter_size = kwargs.pop('filter_size', 3)

        # Other internal params
        input_dim = kwargs.pop('input_dim', (3, 32, 32))
        num_filters = kwargs.pop('num_filters', [16, 32])
        hidden_dims = kwargs.pop('hidden_dims', [100, 100])
        num_classes = kwargs.pop('num_classes', 10)
        self.L = len(num_filters)
        self.M = len(hidden_dims)
        self.num_layers = self.L + self.M + 1
        self.bn_params = {}

        # Internal parameter dict
        self.params = {}

        # Size of the input
        Cinput, Hinput, Winput = input_dim
        stride_conv = 1

        # Init the weights for the conv layers
        F = [Cinput] + num_filters
        for i in range(self.L):
            idx = i + 1
            W = self._weight_init(F[i+1], F[i], fsize=self.filter_size)
            b = np.zeros(F[i+1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(F[i+1]),
                            'running_var':  np.zeros(F[i+1])}
                gamma = np.zeros(F[i + 1])
                beta = np.zeros(F[i + 1])
                self.bn_params.update({
                    'bn_param' + str(idx): bn_param})
                self.params.update({
                    'gamma' + str(idx): gamma,
                    'beta' + str(idx): beta})

        # Init the weights for the affine relu layers
        # Start with the size of the last activation
        Hconv, Wconv = self._size_conv(stride_conv, self.filter_size, Hinput, Winput, self.L)
        dims = [Hconv * Wconv * F[-1]] + hidden_dims
        for i in range(self.M):
            idx = self.L + i + 1
            W = self._weight_init(dims[i], dims[i+1])
            b = np.zeros(dims[i + 1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                             'running_mean': np.zeros(dims[i + 1]),
                             'running_var': np.zeros(dims[i + 1])}
                gamma = np.ones(dims[i + 1])
                beta = np.ones(dims[i + 1])
                self.bn_params.update({
                    'bn_param' + str(idx): bn_param})
                self.params.update({
                    'gamma' + str(idx): gamma,
                    'beta' + str(idx): beta})

        # Scoring layer
        W = self._weight_init(dims[-1], num_classes)
        b = np.zeros(num_classes)
        idx = self.L + self.M + 1
        self.params.update({'W' + str(idx): W,
                            'b' + str(idx): b})

        # Cast parameters to correct type
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def _size_conv(self, stride_conv, filter_size, H, W, n_conv):
        P = int((filter_size - 1)/ 2)
        Hc = int(1+ (H + 2 * P - filter_size) / stride_conv)
        Wc = int(1+ (W + 2 * P - filter_size) / stride_conv)
        wpool = 2   # width
        hpool = 2   # height
        spool = 2   # stride of pool
        Hp = int(1 + (Hc - hpool) / spool)
        Wp = int(1 + (Wc - wpool) / spool)

        if n_conv == 1:
            return Hp, Wp
        else:
            # recursively sub-divide
            H = Hp
            W = Wp
            return self._size_conv(stride_conv, filter_size, H, W, n_conv-1)

    def __str__(self) -> str:
        s = []
        s.append("%d layer network\n" % self.num_layers)
        s.append('weight init : %s\n' % self.weight_init)
        for k, v in self.params.items():
            if k[:1] == 'W':
                w = self.params[k]
                if len(w.shape) == 4:       # conv layer
                    s.append("\t(%d) Conv layer : %s \n" % (int(k[1:]), str(w.shape)))
                else:
                    s.append("\t(%d) FC Layer   : %s \n" % (int(k[1:]), str(w.shape)))

        return ''.join(s)

    def __repr__(self) -> str:
        s = []
        conv_layers = []
        fc_layers = []
        for k in sorted(self.params.keys()):
            if k[:1] == 'W':
                if len(self.params[k].shape) == 4:
                    conv_layers.append('c%d-' % int(self.params[k].shape[0]))
                else:
                    fc_layers.append('fc%d-' % int(self.params[k].shape[1]))
        s.extend(conv_layers)
        s.extend(fc_layers)
        s.extend('f%d' % self.filter_size)
        s.extend('-net')

        return ''.join(s)

    def _weight_init(self, N:int, D:int, fsize=None) -> np.ndarray:
        """
        WEIGHT_INIT
        Set up the weights for a given layer.
        """
        if self.weight_init == 'gauss':
            if fsize is None:
                W = self.weight_scale * np.random.randn(N, D)
            else:
                W = self.weight_scale * np.random.randn(N, D, fsize, fsize)
        elif self.weight_init == 'gauss_sqrt':
            if fsize is None:
                W = self.weight_scale * np.random.randn(N, D) * (1 / np.sqrt(2.0 / (N+D)))
            else:
                W = self.weight_scale * np.random.randn(N, D, fsize, fsize) * (1 / np.sqrt(2.0 / (N+D)))
        elif self.weight_init == 'gauss_sqrt2':
            if fsize is None:
                W = np.random.randn(N, D) * (1 / np.sqrt(2/(N+D)))
            else:
                W = np.random.randn(N, D, fsize, fsize) * (1 / np.sqrt(2/(N+D)))
        elif self.weight_init == 'xavier':
            w_lim = 2 / np.sqrt(N + D)
            if fsize is None:
                wsize = (N, D)
            else:
                wsize = (N, D, fsize, fsize)
            W = np.random.uniform(low=-w_lim, high=w_lim, size=wsize)
        else:
            raise ValueError('Invalid weight init method %s' % self.weight_init)

        return W


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convnet
        """

        X = X.astype(self.dtype)
        N = X.shape[0]
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # Layer parameters
        conv_param = {'stride': 1, 'pad': int((self.filter_size - 1) / 2)}
        pool_param = {'pool_height': 2, 'pool_width': 2,  'stride': 2}
        if self.use_batchnorm:
            for k, bn in self.bn_params.items():
                bn[mode] = mode

        scores = None
        blocks = {}
        blocks['h0'] = X
        # ===============================
        # FORWARD PASS
        # ===============================

        # Forward into conv block
        for l in range(self.L):
            idx = l + 1
            W = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = blocks['h' + str(idx-1)]

            if self.use_batchnorm:
                beta = self.params['beta' + str(idx)]
                gamma = self.params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                h, cache_h = conv_layers.conv_norm_relu_pool_forward(h, W, b,
                                                                conv_param, pool_param,
                                                                gamma, beta, bn_param)
            else:
                h, cache_h = conv_layers.conv_relu_pool_forward(h, W, b, conv_param, pool_param)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        # Forward into linear blocks
        for l in range(self.M):
            idx = self.L + l + 1
            h = blocks['h' + str(idx-1)]
            if l == 0:
                h = h.reshape(N, np.prod(h.shape[1:]))

            W = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_batchnorm:
                beta = self.params['beta' + str(idx)]
                gamma = self.params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                h, cache_h = layers.affine_norm_relu_forward(h, W, b,
                                                             gamma, beta, bn_param)
            else:
                h, cache_h = layers.affine_relu_forward(h, W, b)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        # Forward into the score
        idx = self.L + self.M + 1
        W = self.params['W' + str(idx)]
        b = self.params['b' + str(idx)]
        h = blocks['h' + str(idx-1)]
        h, cache_h = layers.affine_forward(h, W, b)
        blocks['h' + str(idx)] = h
        blocks['cache_h' + str(idx)] = cache_h

        scores = blocks['h' + str(idx)]

        if y is None:
            return scores

        loss = 0.0
        grads = {}
        # Compute the loss
        data_loss, dscores = layers.softmax_loss(scores, y)
        reg_loss = 0.0
        for k in self.params.keys():
            if k[0] == 'W':
                for w in self.params[k]:
                    reg_loss += 0.5 * self.reg * np.sum(w * w)
        loss = data_loss + reg_loss

        # ===============================
        # BACKWARD PASS
        # ===============================
        idx = self.L + self.M + 1
        dh = dscores
        h_cache = blocks['cache_h' + str(idx)]
        dh, dW, db = layers.affine_backward(dh, h_cache)
        blocks['dh' + str(idx-1)] = dh
        blocks['dW' + str(idx)] = dW
        blocks['db' + str(idx)] = db

        # Backprop into the linear layers
        for l in range(self.M)[::-1]:
            idx = self.L + l + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if self.use_batchnorm:
                dh, dW, db, dgamma, dbeta = layers.affine_norm_relu_backward(dh, h_cache)
                blocks['dgamma' + str(idx)] = dgamma
                blocks['dbeta' + str(idx)] = dbeta
            else:
                dh, dW, db = layers.affine_relu_backward(dh, h_cache)
            blocks['dh' + str(idx-1)] = dh
            blocks['dW' + str(idx)] = dW
            blocks['db' + str(idx)] = db

        # Backprop into conv blocks
        for l in range(self.L)[::-1]:
            idx = l + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if l == max(range(self.L)[::-1]):
                dh = dh.reshape(*blocks['h' + str(idx)].shape)

            if self.use_batchnorm:
                dh, dW, db, dgamma, dbeta = conv_layers.conv_norm_relu_pool_backward(dh, h_cache)
                blocks['dgamma' + str(idx)] = dgamma
                blocks['dbeta' + str(idx)] = dbeta
            else:
                dh, dW, db = conv_layers.conv_relu_pool_backward(dh, h_cache)
            blocks['dh' + str(idx-1)] = dh
            blocks['dW' + str(idx)] = dW
            blocks['db' + str(idx)] = db

        # Add reg term to W gradients
        dw_list = {}
        for key, val in blocks.items():
            if key[:2] == 'dW':
                dw_list[key[1:]] = val + self.reg * self.params[key[1:]]

        db_list = {}
        for key, val in blocks.items():
            if key[:2] == 'db':
                db_list[key[1:]] = val

        ## TODO : This is a hack
        dgamma_list = {}
        for key, val in blocks.items():
            if key[:6] == 'dgamma':
                dgamma_list[key[1:]] = val

        # TODO : This is a hack
        dbeta_list = {}
        for key, val in blocks.items():
            if key[:5] == 'dbeta':
                dbeta_list[key[1:]] = val

        grads = {}
        grads.update(dw_list)
        grads.update(db_list)
        grads.update(dgamma_list)
        grads.update(dbeta_list)

        return loss, grads


class ThreeLayerConvNet:
    """
    A three layer convolutional network with the following architecture

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W, and with C input
    channels

    """

    # TODO : Use **kwargs
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=1.0,
                 dtype=np.float32, use_batchnorm=False, verbose=False):
        """
        TODO : Docstring
        """

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.dtype = dtype
        self.weight_scale = weight_scale
        self.bn_params = {}
        self.params = {}
        self.num_layers = 3     # For verbose mode in ex_convnet

        # Init weights and biases for three-layer convnet
        C, W, H = input_dim
        F = num_filters
        # Only use square filters
        Fh = filter_size
        Fw = filter_size
        stride_conv = 1     # TODO : make a settable param
        P = (filter_size - 1 ) / 2  # pad
        Hc = int(1 + (H + 2 * P - Fh) / stride_conv)
        Wc = int(1 + (W + 2 * P - Fw) / stride_conv)

        W1 = self.weight_scale * np.random.randn(F, C, Fh, Fw)
        b1 = np.zeros(F)

        # Pool layer
        # ======== DIMS ========
        # Input - (N, F, Hc, Wc)
        # Output - (N, F, Hp, Wp)

        w_pool = 2
        h_pool = 2
        stride_pool = 2
        Hp = int(1 + (Hc - h_pool) / stride_pool)
        Wp = int(1 + (Wc - w_pool) / stride_pool)

        # Hidden affine layer
        # ======== DIMS ========
        # Input - (N, Fp * Hp * Wp)
        # Output - (N, Hh)
        Hh = hidden_dim
        W2 = self.weight_scale * np.random.randn(F * Hp * Wp, Hh)
        b2 = np.zeros(Hh)

        # Output affine layer
        # ======== DIMS ========
        # Input - (N, Hh)
        # Output - (N, Hc)
        Hc = num_classes
        W3 = self.weight_scale * np.random.randn(Hh, Hc)
        b3 = np.zeros(Hc)

        self.params.update({
            'W1': W1,
            'W2': W2,
            'W3': W3,
            'b1': b1,
            'b2': b2,
            'b3': b3})

        # TODO : batchnorm params

        # Convert datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convnet
        """

        X = X.astype(self.dtype)  # convert datatype
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # TODO: Batchnorm here

        N = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # TODO : more batchnorm stuff here

        fsize = W1.shape[2]
        conv_param = {'stride': 1,
                      'pad': int((fsize - 1) / 2)}
        pool_param = {'pool_height': 2,
                      'pool_width': 2,
                      'stride': 2}
        scores = None

        # ===============================
        # FORWARD PASS
        # ===============================
        x = X
        w = W1
        b = b1
        # Forward into the conv layer
        # TODO : batchnorm
        conv_layer, cache_conv_layer = conv_layers.conv_relu_pool_forward(x, w, b, conv_param, pool_param)

        N, F, Hp, Wp = conv_layer.shape     # Shape of output

        # Forward into the hidden layer
        x = conv_layer.reshape((N, F, Hp * Wp))
        w = W2
        b = b2
        hidden_layer, cache_hidden_layer = layers.affine_relu_forward(x, w, b)
        N, Hh = hidden_layer.shape

        # Forward into linear output layer
        x = hidden_layer
        w = W3
        b = b3
        scores, cache_scores = layers.affine_forward(x, w, b)

        if mode == 'test':
            return scores

        loss = 0
        grads = {}
        # ===============================
        # BACKWARD PASS
        # ===============================
        data_loss, dscores = layers.softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss = 0.5 * self.reg * np.sum(W2**2)
        reg_loss = 0.5 * self.reg * np.sum(W3**2)
        loss = data_loss + reg_loss

        # backprop into output layer
        dx3, dW3, db3 = layers.affine_backward(dscores, cache_scores)
        dW3 += self.reg * W3

        # backprop into first fc layer
        dx2, dW2, db2 = layers.affine_relu_backward(dx3, cache_hidden_layer)
        dW2 += self.reg * W2

        # Backprop into conv layer
        dx2 = dx2.reshape(N, F, Hp, Wp)           # Note - don't forget to reshape here...
        dx, dW1, db1 = conv_layers.conv_relu_pool_backward(dx2, cache_conv_layer)
        dW1 += self.reg * W1

        grads.update({
            'W1': dW1,
            'W2': dW2,
            'W3': dW3,
            'b1': db1,
            'b2': db2,
            'b3': db3})

        return loss, grads
