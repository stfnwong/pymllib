"""
CAPTIONING SOLVER
Encapsulates logic required for training image captioning models

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import numpy as np
import pickle
from pymllib.solver import optim
from pymllib.utils import coco_utils

# debug
#from pudb import set_trace; set_trace()

# TODO : Debug function for model types
def print_ptypes(model):
    for k, v in model.params.items():
        print('%s : %s' % (k, type(v)))

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
        # Debug, print
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        # Checkpoints
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', None)

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
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _get_checkpoint(self):
        checkpoint = {
            # Model data
            'model': self.model,
            # Solver params
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'num_epochs': self.num_epochs,
            # Solution data
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
            # Loss window
            #'enable_loss_window': self.enable_loss_window,
            #'loss_window_len': self.loss_window_len,
            #'loss_window_eps': self.loss_window_eps,
            #'loss_converge_window': self.loss_converge_window,
            # Checkpoint info
            'checkpoint_name': self.checkpoint_name,
            'checkpoint_dir': self.checkpoint_dir
        }

        return checkpoint


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


    def load_checkpoint(self, fname):
        """
        LOAD_CHECKPOINT
        Load a saved checkpoint from disk into a solver object.
        In the current version of this method defaults are provided for
        missing attributes. This somewhat obviates the need to have a
        conversion utility, as the such a utility would be inserting
        dummy values into attributes that are missing anyway.
        """

        with open(fname, 'rb') as fp:
            cpoint_data = pickle.load(fp)

        # Model data
        self.model = cpoint_data.get('model')
        # Solver params
        self.update_rule = cpoint_data.get('update_rule')
        self.lr_decay = cpoint_data.get('lr_decay')
        self.optim_config = cpoint_data.get('optim_config')
        self.batch_size = cpoint_data.get('batch_size')
        self.epoch = cpoint_data.get('epoch')
        self.num_epochs = cpoint_data.get('num_epochs', 0)
        # Solution data
        self.loss_history = cpoint_data.get('loss_history')
        #self.train_acc_history = cpoint_data.get('train_acc_history')
        #self.val_acc_history = cpoint_data.get('val_acc_history')
        ## Loss window
        #self.enable_loss_window = cpoint_data.get('enable_loss_window', False)
        #self.loss_window_len = cpoint_data.get('loss_window_len', 500)
        #self.loss_window_eps = cpoint_data.get('loss_window_eps', 1e-4)
        #self.loss_converge_window = cpoint_data['loss_converge_window']

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

            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                self._save_checkpoint()

            # At the end of each epoch decay learning rate,
            # increment epoch counter
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # TODO : check accuracy
