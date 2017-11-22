"""
SOLVER_UTILS
Utils to load solvers and checkpoints from disk individually or in batches

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pymllib.solver.solver as solver

# Debug
#from pudb import set_trace; set_trace()

def examine_checkpoint(fname, verbose=False):
    """
    EXAMINE_CHECKPOINT
    Print some relevant data about a checkpoint, without
    loading it into a solver object
    """

    with open(fname, 'rb') as fp:
        cpoint = pickle.load(fp)

    if verbose is True:
        print('File : %s' % fname)

    for k, v in cpoint.items():
        if k == 'train_acc_history' or k == 'val_acc_history' or k == 'loss_history':
            print("%s : length %d" % (k, len(cpoint[k])))
        else:
            print('%s : %s' % (k, v))


def convert_checkpoint(fname, verbose=False):
    """
    Convert a checkpoint to the newest version.
    This method is designed to convert old checkpoints to the
    current format, which unifies the save and load methods to that
    there is no difference between saving a solver and saving a
    checkpoint. Because there may be checkpoints from previous
    versions lying around that may break when loaded, this tool will
    convert old checkpoints so that this doesnt occur
    """
    solv = solver.Solver(None, None)
    solv.verbose = verbose
    #solv.load(fname)

    with open(fname, 'rb') as fp:
        cpoint = pickle.load(fp)

        # Model data
        solv.model = cpoint.get('model', None)
        # Solver params
        solv.update_rule = cpoint.get('update_rule', 'sgd')
        solv.lr_decay = cpoint.get('lr_decay', 0.95)
        solv.optim_config = cpoint.get('optim_config', {'learning_rate': 1e-3})
        solv.batch_size = cpoint.get('batch_size', 100)
        solv.epoch = cpoint.get('epoch', 0)
        solv.num_epochs = cpoint.get('num_epochs', 0)
        # Solution data
        solv.loss_history = cpoint.get('loss_history', None)
        solv.train_acc_history = cpoint.get('train_acc_history', None)
        solv.val_acc_history = cpoint.get('val_acc_history', None)
        # Loss window
        solv.enable_loss_window = cpoint.get('enable_loss_window', False)
        solv.loss_window_len = cpoint.get('loss_window_len', 500)
        solv.loss_window_eps = cpoint.get('loss_window_eps', 1e-3)
        solv.loss_converge_window = cpoint.get('loss_converge_window', 1e4)
        # Checkpoint info
        solv.checkpoint_name = cpoint.get('checkpoint_name', None)
        solv.checkpoint_dir = cpoint.get('checkpoint_dir', None)

    # This solver has now been 'converted'
    return solv
