"""
TEST_SOLVER
Test the solver object and the various optimization functions
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import matplotlib.pyplot as plt
import numpy as np
import error
import data_utils
# Units under test
import solver
import optim
import fcnet

import unittest


def get_figure_handles():
    fig = plt.figure()
    ax = []
    for i in range(3):
        sub_ax = fig.add_subplot(3,1,(i+1))
        ax.append(sub_ax)

    return fig, ax

# Show the solver output
def plot_test_result(ax, solver_dict):

    assert len(ax) == 3, "Need 3 axis"

    for n in range(len(ax)):
        ax[n].set_xlabel("Epoch")
        if n == 0:
            ax[n].set_title("Training Loss")
        elif n == 1:
            ax[n].set_title("Training Accuracy")
        elif n == 2:
            ax[n].set_title("Validation Accuracy")

    # update data
    for method, solv in solver_dict.items():
        ax[0].plot(solv.loss_history, 'o', label=method)
        ax[1].plot(solv.train_acc_history, '-x', label=method)
        ax[2].plot(solv.val_acc_history, '-x', label=method)

    # Update legend
    for i in range(len(ax)):
        ax[i].legend(loc='upper right', ncol=4)
    # Note: outside the function we set
    # fig.set_size_inches(8,8)
    # fig.tight_layout()


def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-6
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.verbose = False

    def test_sgd_momentum(self):
        print("\n======== TestSolver.test_sgd_momentum:")

        N = 4
        D = 5

        w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
        v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

        config = {'learning_rate': 1e-3, 'velocity': v}
        next_w, _ = optim.sgd_momentum(w, dw, config=config)
        expected_next_w = np.asarray([
        [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
        [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
        [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
        [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
        expected_velocity = np.asarray([
        [ 0.5406,      0.55475789,  0.56891579,  0.58307368,  0.59723158],
        [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
        [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
        [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])

        next_w_error = error.rel_error(next_w, expected_next_w)
        velocity_error = error.rel_error(config['velocity'], expected_velocity)

        print("next_w_error = %f" % next_w_error)
        print("velocity_error = %f" % velocity_error)

        self.assertLessEqual(next_w_error, self.eps)
        self.assertLessEqual(velocity_error, self.eps)

        print("======== TestSolver.test_sgd_momentum: <END> ")

    def test_all_optim(self):
        print("\n======== TestSolver.test_all_optim:")

        dataset =  load_data(self.data_dir, self.verbose)

        #optim_list = ['sgd', 'sgd_momentum', 'rmsprop']
        optim_list = ['sgd', 'sgd_momentum']
        num_train = 50

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        #hidden_dims = [100, 100, 100, 100, 100]
        hidden_dims = [100, 50, 10]     # just some random dims
        weight_scale = 5e-2
        learning_rate = 1e-2
        num_epochs = 20
        batch_size = 50
        solvers = {}

        for update_rule in optim_list:
            print("Using update rule %s" % update_rule)
            model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            dtype=np.float64)
            if self.verbose:
                print(model)
            model_solver = solver.Solver(model,
                                        small_data,
                                        print_every=100,
                                        num_epochs=num_epochs,
                                        batch_size=batch_size,     # previously 25
                                        update_rule=update_rule,
                                        optim_config={'learning_rate': learning_rate})
            solvers[update_rule] = model_solver
            model_solver.train()

        # get some figure handles and plot the data
        fig,ax = get_figure_handles()
        plot_test_result(ax, solvers)
        fig.set_size_inches(8,8)
        fig.tight_layout()
        plt.show()

        print("======== TestSolver.test_all_optim: <END> ")


if __name__ == "__main__":
    unittest.main()
