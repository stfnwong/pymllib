"""
EX_SOLVER_UTILS
Test the new solver utils

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymllib.utils import solver_utils


if __name__ == "__main__":
    prefix = '/home/kreshnik/Documents/compucon/machine-learning/models'
    path = 'conv-net-train-cifar-stage-200-epoch'
    fname = 'c16-c32-c64-c128-fc256-fc256-fc10--f3-net'

    num_epochs = 200
    for n in range(num_epochs):
        suffix = '_epoch_%d.pkl' % int(n+1)
        cname = '%s/%s/%s%s' % (str(prefix), str(path), str(fname), str(suffix))
        solver_utils.examine_checkpoint(cname, verbose=True)
        solv = solver_utils.convert_checkpoint(cname)
        solv.save(cname)        # write over old checkpoint
