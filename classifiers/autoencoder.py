"""
AUTOENCODER
Attempt to use an fcnet as an autoencoder

Stefan Wong 2017
"""


def sparse_autoencoder_loss(scores, y):
    pass

class Autoencoder(object):
    def __init__(self, hidden_dims, input_dim, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,
                 verbose=False):

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Init the params of the network into the dictionary self.params
        dims = [input_dim] + hidden_dims + [num_classes]
        Ws = {'W' + str(i+1) : weight_scale * np.random.randn(dims[i], dims[i+1]) for i in range(len(dims)-1)}
        bs = {'b' + str(i+1) : np.zeros(dims[i+1]) for i in range(len(dims)-1)}
        self.params.update(bs)
        self.params.update(Ws)

        # Cast params
        if self.verbose:
            print("Casting parameers to type %s" % self.dtype)
        for k,v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def __str__(self):
        s = []
        for l in range(self.num_layers):
            wl = self.params['W' + str(l+1)]
            bl = self.params['b' + str(l+1)]
            s.append('Layer %d\n\t W%d: (%d, %d),\t b%d: (%d)\n' % (l+1, l+1, wl.shape[0], wl.shape[1], l+1, bl.shape[0]))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()


    def loss(self, X, y=None):

        X = X.astype(self.dtype)

        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # ===============================
        # FORWARD PASS
        # ===============================
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))   # TODO ; Check this...


        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_dropout:
                h = hidden['hdrop' + str(idx-1)]
            else:
                h = hidden['h' + str(idx-1)]

            # Compute the forward pass
            # output layer is a special case
            if idx == self.num_layers:
                h, cache_h = layers.affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            else:
                h, cache_h = layers.affine_relu_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h

        scores = hidden['h' + str(self.num_layers)]

        if mode == 'test':
            return scores

        loss = 0.0
        grads = {}
        # Compute loss
        # Here we don't want to use the softmax loss, rather
        data_loss, dscores = sparse_autoencoder_loss(scores, y)
        reg_loss = 0
        for f in self.params.keys():
            if f[0] == 'W':
                for w in self.params[f]:
                    reg_loss += 0.5 * self.reg * np.sum(w * w)

