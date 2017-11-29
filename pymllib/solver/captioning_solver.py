"""
CAPTIONING SOLVER
Encapsulates logic required for training image captioning models

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import numpy as np
from pymllib.solver import optim
from pymllib.utils import coco_utils

# debug
#from pudb import set_trace; set_trace()

class CaptioningSolver(object):
    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        """
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        # Debug, pring
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Keep reference to data and model
        self.model = model
        self.data = data

        # Check for extra keyword args
        if len(kwargs) > 0:
            extra = ', '.join('"%s" ' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exist, then replace the string
        # name with a reference to the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update rule %s' % self.update_rule)

        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Do some internal bookkeping
        """

        self.epoch = 0.0
        self.best_val_acc = 0.0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim config
        # for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update
        """

        # create a minibatch of data
        minibatch = coco_utils.sample_coco_minibatch(self.data,
                                    batch_size=self.batch_size,
                                    split='train')
        captions, features, urls = minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # Perform parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w,
            self.optim_configs[p] = next_config


    def train(self):
        """
        Run optimization to train the model

        """
        # TODO : Add the loss window stuff ? Or perhaps take
        # the loss window out of the other branches....

        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = int(max(num_train / self.batch_size, 1))
        num_iterations = int(self.num_epochs * iterations_per_epoch)

        for t in range(num_iterations):
            self._step()
            # print loss
            if self.verbose and t % self.print_every == 0:
                print("(Iteration %d / %d) loss : %f" % (t+1, num_iterations, self.loss_history[-1]))

            # At the end of each epoch decay learning rate,
            # increment epoch counter
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # TODO : check accuracy

