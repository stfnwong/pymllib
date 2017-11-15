"""
SOLVER
A solver object that performs training on a neural network. Taken from
Caffe/CS231n

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))

import numpy as np
import pickle
import pymllib.solver.optim as optim

# Debug
#from pudb import set_trace; set_trace()

class Solver(object):
    """
    TODO : Docstring
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new solver instance

        TODO : Rest of docstring
        """

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
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', 'checkpoint')
        # The idea here is that if the loss doesn't change by eps for more than
        # 500 iters we quit
        self.loss_window_len = kwargs.pop('loss_window_len', 500)
        self.loss_window_eps = kwargs.pop('loss_window_eps', 1e-3)
        self.loss_converge_window = kwargs.pop('loss_converge_window', 1e4)

        if model is None or data is None:
            # assume we are loading from file
            self.model = None
            self.X_train = None
            self.y_train = None
            self.X_val = None
            self.y_val = None

            return

        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

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

    def __str__(self):
        s = []
        # print the size of the dataset attached to the solver
        #s.append("X_train shape  (%s)" % str(self.X_train.shape))
        #s.append("y_trian shape  (%s)" % str(self.y_train.shape))
        #s.append("X_val shape    (%s)" % str(self.X_val.shape))
        #s.append("y_val shape    (%s)" % str(self.y_val.shape))
        # Solver params
        s.append("update rule  : %s\n" % str(self.update_rule))
        s.append("optim config : %s\n" % str(self.optim_config))
        s.append("lr decay     : %s\n" % str(self.lr_decay))
        s.append("batch size   : %s\n" % str(self.batch_size))
        s.append("num epochs   : %s\n" % str(self.num_epochs))
        s.append("print every  : %d\n" % self.print_every)

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

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
            d = {k: v for k, v in self.optim_config.items()}
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
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _get_checkpoint(self):
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            #'num_train_samples': self.num_train_samples,
            #'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }

        return checkpoint

    def _get_solver_params(self):
        params = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'print_every': self.print_every,
            #'num_train_samples': self.num_train_samples,
            #'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }

        return params

    def _save_checkpoint(self):
        """
        Save the current training status
        """

        if self.checkpoint_name is None:
            return

        checkpoint = self._get_checkpoint()
        filename = "%s/%s_epoch_%d.pkl" % (self.checkpoint_dir, self.checkpoint_name, self.epoch)
        if self.verbose:
            print("Saving checkpoint to file %s" % filename)
        with open(filename, 'wb') as fp:
            pickle.dump(checkpoint, fp)

    # TODO: should there be a _load_checkpoint()?

    def save(self, filename):
        params = self._get_solver_params()

        if self.verbose:
            print("Saving model to file %s" % filename)
        with open(filename, 'wb') as fp:
            pickle.dump(params, fp)

    def load(self, filename):
        """
        Load an entire model from disk
        """

        if self.verbose:
            print("Loading model from file %s" % filename)

        with open(filename, 'rb') as fp:
            model_data = pickle.load(fp)
            # Copy the params to this object
            self.model = model_data['model']
            self.update_rule = model_data['update_rule']
            self.lr_decay = model_data['lr_decay']
            self.optim_confi = model_data['optim_config']
            self.batch_size = model_data['batch_size']
            self.num_epochs = model_data['num_epochs']
            self.print_every = model_data['print_every']
            #self.num_train_samples = model_data['num_train_samples']
            #self.num_val_samples = model_data['num_val_samples']
            self.epoch = model_data['epoch']
            self.loss_history = model_data['loss_history']
            self.train_acc_history = model_data['train_acc_history']
            self.val_acc_history = model_data['val_acc_history']


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

        num_batches = int(N / batch_size)
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
        num_iterations = int(self.num_epochs * iterations_per_epoch)
        # Setup window to compute minimum loss over
        if self.loss_window_len > num_iterations:
            loss_win = num_iterations
        else:
            loss_win = self.loss_window_len
        avg_loss = 0.0
        prev_avg_loss = 0.0

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
                self._save_checkpoint()

                if self.verbose:
                    print("[Epoch %6d/%6d] train acc : %f, val acc: %f" % (self.epoch, self.num_epochs, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

            # See if the loss has changes sufficiently
            if t > loss_win:
                avg_loss = sum(self.loss_history[-loss_win:]) / loss_win
                if abs(avg_loss - prev_avg_loss) < self.loss_window_eps:
                    if self.verbose:
                        print("Difference has changed by less than %f in %d iterations, exiting\n" % (self.loss_window_eps, t))
                    return
                prev_avg_loss = avg_loss

        # Swap the best parameters into the model
        self.model.params = self.best_params
