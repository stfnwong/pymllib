"""
SOLVER
A solver object that performs training on a neural network. Taken from
Caffe/CS231n

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))

from solver import optim
import numpy as np


class Solver(object):
    """
    TODO : Docstring
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new solver instance

        TODO : Rest of docstring
        """

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword args
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 0.95)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.check_point_every = kwargs.pop('check_point_every', 1)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Make sure there are no additional arguments
        if len(kwargs) > 0:
            extra = ''.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % (extra))

        # Make sure the update rule exists, then replace string
        # with actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update rule "%s"' % (self.update_rule))
        self.update_rule = getattr(optim, self.update_rule)
        self._reset()

    def _reset(self):
        """
        Set up some bookkeeping variables for optimization
        """

        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of optim for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update
        """

        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute the loss, gradients
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.iteritems():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.optim_configs[p] = next_config

    # TODO : Come back to this last
    #def save_checkpoint(self):
    #    """
    #    Save the current training status
    #    """

    #    if self.checkpoint_name is None:
    #        return

    #    checkpoint = {
    #        'model': self.model,
    #        'update_rule': self.update_rule,
    #        'lr_decay': self.lr_decay,
    #        'optim_config': self.optim_config,
    #        'batch_size': self.batch_size,
    #        'num_train_samples': self.num_train_samples,
    #        'num_val_samples': self.num_train_samples,


    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
            - X : Array of data of shape (N, d_1, ..., d_k)
            - y : Array of labels of shape (N,)
            - num_samples : If not None, subsample the data and only
            test the model on num_samples datapoints.
            - batch_size: Split X and y into batches of this size to
            avoid using too much memory
        """

        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches +=1
        y_pred = []

        for i in range(num_batches):
            start = i * batch_size
            end = (i+1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)      # Stacks arrays in sequence column-wise
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model
        """

        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):

            self._step()

            # Print training loss
            if self.verbose and (t % self.print_every == 0):
                print("[Iteration %6d/%6d] loss : %f" % (t+1, num_iterations, self.loss_history[-1]))

            # Increment epoch counter, decay learning rate
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on first iteration, last iteration,
            # and at the end of each epoch
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print("[Epoch %6d/%6d] train acc : %f, val acc: %f" % (self.epoch, self.num_epochs, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.iteritems():
                        self.best_params[k] = v.copy()

        # Swap the best parameters into the model
        self.model.params = self.best_params
