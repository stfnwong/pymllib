"""
SOLVER
A solver object that performs training on a neural network. Taken from
Caffe/CS231n

Stefan Wong 2017
"""

import numpy as np
import pickle
from pymllib.solver import optim

from typing import Any
from typing import Dict
from typing import List
from typing import Union

# Debug
#from pudb import set_trace; set_trace()

class Solver:
    """
    TODO : Docstring
    """

    def __init__(self, model:Any, data:Union[Dict[str, Any], None], **kwargs) -> None:
        """
        Construct a new solver instance

        TODO : Rest of docstring
        """
        self.model   :Any        = model
        self.X_train :np.ndarray = data['X_train']
        self.y_train :np.ndarray = data['y_train']
        self.X_val   :np.ndarray = data['X_val']
        self.y_val   :np.ndarray = data['y_val']

        # Unpack keyword args
        self.update_rule  :str            = kwargs.pop('update_rule', 'sgd')
        self.optim_config :Dict[str, Any] = kwargs.pop('optim_config', {})
        self.lr_decay     :float          = kwargs.pop('lr_decay', 0.95)
        self.batch_size   :float          = kwargs.pop('batch_size', 100)
        self.num_epochs   :float          = kwargs.pop('num_epochs', 10)

        self.check_point_every :int  = kwargs.pop('check_point_every', 1)
        self.checkpoint_name   :str  = kwargs.pop('checkpoint_name', None)
        self.print_every       :int  = kwargs.pop('print_every', 10)
        self.verbose           :bool = kwargs.pop('verbose', True)
        self.checkpoint_dir    :str  = kwargs.pop('checkpoint_dir', 'checkpoint')
        # The idea here is that if the loss doesn't change by eps for more than
        # 500 iters we quit
        self.enable_loss_window = kwargs.pop('enable_loss_window', False)
        self.loss_window_len    = kwargs.pop('loss_window_len', 500)
        self.loss_window_eps    = kwargs.pop('loss_window_eps', 1e-3)
        #self.loss_converge_window = kwargs.pop('loss_converge_window', 1e4)

        if model is None or data is None:
            # assume we are loading from file
            self.model   :Any= None
            self.X_train :Union[np.ndarray, None] = None
            self.y_train :Union[np.ndarray, None] = None
            self.X_val   :Union[np.ndarray, None] = None
            self.y_val   :Union[np.ndarray, None] = None

            return

        self.model   :Any        = model
        self.X_train :np.ndarray = data['X_train']
        self.y_train :np.ndarray = data['y_train']
        self.X_val   :np.ndarray = data['X_val']
        self.y_val   :np.ndarray = data['y_val']

        # Make sure there are no additional arguments
        if len(kwargs) > 0:
            extra = ''.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % (extra))

        # Make sure the update rule exists, then replace string
        # with actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update rule "%s"' % (self.update_rule))
        self.update = getattr(optim, self.update_rule)()       # TODO: is this not the correct syntax for this?
        self._reset()

    def __str__(self) -> str:
        s = []
        if self.X_train is not None:
            # print the size of the dataset attached to the solver
            s.append("Data shape:\n")
            s.append("X_train shape  (%s)\n" % str(self.X_train.shape))
            s.append("y_train shape  (%s)\n" % str(self.y_train.shape))
            s.append("X_val shape    (%s)\n" % str(self.X_val.shape))
            s.append("y_val shape    (%s)\n" % str(self.y_val.shape))
        # Solver params
        s.append("Solver parameters:\n")
        s.append("update rule  : %s\n" % str(self.update_rule))
        s.append("optim config : %s\n" % str(self.optim_config))
        s.append("lr decay     : %s\n" % str(self.lr_decay))
        s.append("batch size   : %s\n" % str(self.batch_size))
        s.append("num epochs   : %s\n" % str(self.num_epochs))
        s.append("print every  : %d\n" % self.print_every)
        # Loss window
        if self.enable_loss_window:
            s.append("Loss window:\n")
            s.append("len          : %d\n" % self.loss_window_len)
            s.append("eps          : %d\n" % self.loss_window_eps)

        return ''.join(s)

    def __repr__(self) -> str:
        return self.__str__()

    def _reset(self) -> None:
        """
        Set up some bookkeeping variables for optimization
        """
        self.epoch :int = 0
        self.best_val_acc :float = 0
        self.best_params : Dict[str, Any]= {}
        self.loss_history :List[float] = []
        self.train_acc_history :List[float] = []
        self.val_acc_history :List[float] = []
        # Make a deep copy of optim for each parameter
        self.optim_configs :Dict[str, Any] = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self) -> None:
        """
        Make a single gradient update
        """
        num_train  = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch    = self.X_train[batch_mask]
        y_batch    = self.y_train[batch_mask]

        # Compute the loss, gradients
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _get_checkpoint(self) -> Dict[str, Any]:
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
            'print_every' : self.print_every,
            # Solution data
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
            # Loss window
            'enable_loss_window': self.enable_loss_window,
            'loss_window_len': self.loss_window_len,
            'loss_window_eps': self.loss_window_eps,
            #'loss_converge_window': self.loss_converge_window,
            # Checkpoint info
            'checkpoint_name': self.checkpoint_name,
            'checkpoint_dir': self.checkpoint_dir
        }

        return checkpoint

    def _save_checkpoint(self) -> None:
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

    def load_checkpoint(self, fname:str) -> None:
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
        self.train_acc_history = cpoint_data.get('train_acc_history')
        self.val_acc_history = cpoint_data.get('val_acc_history')
        # Loss window
        self.enable_loss_window = cpoint_data.get('enable_loss_window', False)
        self.loss_window_len = cpoint_data.get('loss_window_len', 500)
        self.loss_window_eps = cpoint_data.get('loss_window_eps', 1e-4)
        #self.loss_converge_window = cpoint_data['loss_converge_window']

    def save(self, filename:str) -> None:
        params = self._get_checkpoint()

        if self.verbose:
            print("Saving model to file %s" % filename)
        with open(filename, 'wb') as fp:
            pickle.dump(params, fp)

    def load(self, filename:str) -> None:
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

    def check_accuracy(self,
                       X:np.ndarray,
                       y:np.ndarray,
                       num_samples:Union[None, int]=None,
                       batch_size:int=100) -> float:
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

    def train(self) -> None:
        """
        Run optimization to train the model
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)

        if self.enable_loss_window:
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

            # Scale epsilon down as the learning rate goes down
            if self.enable_loss_window and epoch_end:
                self.loss_window_eps *= self.lr_decay

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
            if self.enable_loss_window:
                if t > loss_win:
                    avg_loss = sum(self.loss_history[-loss_win:]) / loss_win
                    if abs(avg_loss - prev_avg_loss) < self.loss_window_eps:
                        if self.verbose:
                            print("Difference has changed by less than %f in %d iterations, exiting\n" % (self.loss_window_eps, t))
                        return
                    prev_avg_loss = avg_loss

        # Swap the best parameters into the model
        self.model.params = self.best_params
