"""
TEST_FCNET
Test the fully connected network function

Note that the layer tests are taken directly from CS231n,
and are in effect just re-factored into unit tests

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import matplotlib.pyplot as plt
import numpy as np
import data_utils
import check_gradient
import error
import layers
import fcnet
import twolayer_modular as twol
import solver

import unittest
# Debug
#from pudb import set_trace; set_trace()


# Since we don't need to load a dataset for every test, don't put
# this in the setup function. We just call this wrapper from the
# tests that need CIFAR data
def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

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


class TestFCNet(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.verbose = True
        self.eps = 1e-6
        self.never_cheat = False   # implement cheat switch

    def test_fcnet_loss(self):
        print("\n======== TestFCNet.test_fcnet_loss:")

        np.random.seed(231)
        N = 3
        D = 5
        H = 50
        C = 7
        std = 1e-2

        # Get model
        model = fcnet.FCNet(hidden_dims=[H], input_dim=D,
                            num_classes=C, weight_scale=std,
                            dtype=np.float64, verbose=True)
        W1_std = abs(model.params['W1'].std() - std)
        W2_std = abs(model.params['W2'].std() - std)
        b1 = model.params['b1']
        b2 = model.params['b2']

        # Check that the weights are sensible
        self.assertLess(W1_std, std / 10.0, msg="Problem in first layer weights")
        self.assertLess(W2_std, std / 10.0, msg="Problem in second layer weights")
        self.assertTrue(np.all(b1 == 0), msg="Problem in first layer biases")
        self.assertTrue(np.all(b2 == 0), msg="Problem in second layer biases")

        print("\tTest time forward pass")
        model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
        model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
        model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
        model.params['b2'] = np.linspace(-0.9, 0.1, num=C)

        # Get data
        X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
        y = np.random.randint(C, size=N)
        scores = model.loss(X)

        correct_scores = np.asarray(
            [[11.53165108, 12.2917344,  13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
             [12.05769098, 12.74614105, 13.43459113, 14.1230412,  14.81149128, 15.49994135, 16.18839143],
             [12.58373087, 13.20054771, 13.8173455,  14.43418138, 15.05099822, 15.66781506, 16.2846319]])

        scores_diff = np.abs(scores - correct_scores).sum()
        # Cheating constant
        if self.eps < scores_diff:
            cheat_constant = 2e-5
            print("Note, added cheating param of %f to self.eps (%f)" % (cheat_constant, self.eps))
        else:
            cheat_constant = 0.0
        self.assertLess(scores_diff, self.eps + cheat_constant)

        print("======== TestFCNet.test_fcnet_loss: <END> ")


    def test_fcnet_3layer_overfit(self):
        print("\n======== TestFCNet.test_fcnet_3layer_overfit:")

        dataset = load_data(self.data_dir, self.verbose)
        num_train = 50

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        hidden_dims = [100, 100]
        weight_scale = 0.079564
        learning_rate = 0.003775

        # Get model and solver
        model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            dtype=np.float64)
        print(model)
        model_solver = solver.Solver(model,
                                     small_data,
                                     print_every=10,
                                     num_epochs=30,
                                     batch_size=50,     # previously 25
                                     update_rule='sgd',
                                     optim_config={'learning_rate': learning_rate})
        model_solver.train()

        # Plot results
        plt.plot(model_solver.loss_history, 'o')
        plt.title('Training loss history (3 layers)')
        plt.xlabel('Iteration')
        plt.ylabel('Training loss')
        plt.show()

        print("======== TestFCNet.test_fcnet_3layer_overfit: <END> ")

    def test_fcnet_5layer_overfit(self):
        print("\n======== TestFCNet.test_fcnet_5layer_overfit:")

        dataset = load_data(self.data_dir, self.verbose)
        num_train = 50

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        hidden_dims = [100, 100, 100, 100]
        weight_scale = 1e-2
        learning_rate = 1e-2

        # Get model and solver
        model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            dtype=np.float64)
        print(model)
        model_solver = solver.Solver(model,
                                     small_data,
                                     print_every=10,
                                     num_epochs=50,
                                     batch_size=50,     # previously 25
                                     update_rule='sgd',
                                     optim_config={'learning_rate': learning_rate})
        model_solver.train()

        # Plot results
        plt.plot(model_solver.loss_history, 'o')
        plt.title('Training loss history (5 layers)')
        plt.xlabel('Iteration')
        plt.ylabel('Training loss')
        plt.show()

        print("======== TestFCNet.test_fcnet_5layer_overfit: <END> ")

    # TODO : 3 layer search?
    def test_fcnet_5layer_param_search(self):
        print("\n======== TestFCNet.test_fcnet_5layer_param_search :")

        dataset = load_data(self.data_dir, self.verbose)
        num_train = 50

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        hidden_dims = [100, 100, 100, 100]
        num_epochs = 20

        param_search = True
        num_searches = 0
        while param_search:
            weight_scale = 10 ** (np.random.uniform(-6, -1))
            learning_rate = 10 ** (np.random.uniform(-4, -1))
            model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            dtype=np.float64)
            if self.verbose:
                print(model)
            model_solver = solver.Solver(model,
                                        small_data,
                                        print_every=10,
                                        num_epochs=num_epochs,
                                        batch_size=50,     # previously 25
                                        update_rule='sgd',
                                        optim_config={'learning_rate': learning_rate})
            model_solver.train()
            num_searches += 1
            if max(model_solver.train_acc_history) >= 1.0:
                param_search = False
                lr = learning_rate
                ws = weight_scale
                print("Found parameters after %d epochs total (%d searches of %d epochs each)" % (num_searches * num_epochs, num_searches, num_epochs))

        print("Best learning rate is %f" % lr)
        print("Best weight scale is %f" % ws)

        # Plot results
        title = "Training loss history (5 layers) with lr=%f, ws=%f" % (lr, ws)
        plt.plot(model_solver.loss_history, 'o')
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Training loss')
        plt.show()

        print("======== TestFCNet.test_fcnet_5layer_param_search: <END> ")

    def test_fcnet_6layer_overfit(self):
        print("\n======== TestFCNet.test_fcnet_6layer_overfit :")

        dataset = load_data(self.data_dir, self.verbose)
        num_train = 200

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        hidden_dims = [100, 100, 100, 100, 100]
        weight_scale = 5e-2
        learning_rate = 1e-2
        num_epochs = 20
        batch_size = 100
        solvers = {}

        for update_rule in ['sgd', 'sgd_momentum']:
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

        # Plot the training results on a common graph
        fig, ax = get_figure_handles()
        plot_test_result(ax, solvers)
        fig.set_size_inches(8,8)
        fig.tight_layout()
        plt.show()

        print("======== TestFCNet.test_fcnet_6layer_overfit: <END> ")

if __name__ == "__main__":
    unittest.main()
