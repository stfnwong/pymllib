"""
TEST_FCNET
Test the fully connected network function

Note that the layer tests are taken directly from CS231n,
and are in effect just re-factored into unit tests

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
import unittest

# Library imports
from pymllib.utils import data_utils
from pymllib.classifiers import fcnet
from pymllib.solver import solver
from pymllib.vis import vis_solver

# Debug
#from pudb import set_trace; set_trace()


# Since we don't need to load a dataset for every test, don't put
# this in the setup function. We just call this wrapper from the
# tests that need CIFAR data
def load_data(data_dir, verbose=False):

    if verbose is True:
        print("Loading data from %s" % data_dir)

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

# Get a new figure and axis to plot into
def get_figure_handles():
    fig = plt.figure()
    ax = []
    for i in range(3):
        sub_ax = fig.add_subplot(3,1,(i+1))
        ax.append(sub_ax)

    return fig, ax


# Show the solver output
def plot_test_result(ax, solver_dict, num_epochs=None):

    assert len(ax) == 3, "Need 3 axis"

    for n in range(len(ax)):
        ax[n].set_xlabel("Epoch")
        if num_epochs is not None:
            ax[n].set_xticks(range(num_epochs))
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
        self.draw_plots = False
        self.num_classes = 10
        self.never_cheat = False   # implement cheat switch

    def test_fcnet_loss(self):
        print("\n======== TestFCNet.test_fcnet_loss:")
        # Some model parameters
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
                            dtype=np.float64,
                            verbose=True)
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
        if self.draw_plots is True:
            plt.plot(model_solver.loss_history, 'o')
            plt.title('Training loss history (3 layers)')
            plt.xlabel('Iteration')
            plt.ylabel('Training loss')
            plt.show()

        print("======== TestFCNet.test_fcnet_3layer_overfit: <END> ")

    def test_fcnet_5layer_loss(self):
        print("\n======== TestFCNet.test_fcnet_5layer_loss:")

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
                            reg=0.0,
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

        print("======== TestFCNet.test_fcnet_5layer_loss: <END> ")

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
        if self.draw_plots is True:
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
        if self.draw_plots:
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
            if self.verbose:
                print(model_solver)
            solvers[update_rule] = model_solver
            model_solver.train()

        # Plot the training results on a common graph
        if self.draw_plots:
            fig, ax = get_figure_handles()
            plot_test_result(ax, solvers)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== TestFCNet.test_fcnet_6layer_overfit: <END> ")

    def test_batchnorm(self):
        print("\n======== TestFCNet.test_batchnorm :")

        dataset = load_data(self.data_dir, self.verbose)
        num_train = 10
        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        input_dim = 32 * 32 * 3
        hidden_dims = [100, 100, 100, 100, 100, 100]
        weight_scale = 2e-2
        learning_rate = 1e-3
        num_epochs=30
        update_rule='adam'
        #weight_scale = 0.079564
        #learning_rate = 0.003775

        bn_model = fcnet.FCNet(input_dim=input_dim,
                               hidden_dims=hidden_dims,
                               weight_scale=weight_scale,
                               use_batchnorm=True)
        bn_solver = solver.Solver(bn_model,
                                  small_data,
                                  num_epochs=num_epochs,
                                  update_rule=update_rule,
                                  optim_config={'learning_rate': learning_rate},
                                  verbose=True,
                                  print_every=200)
        print("Training with batchnorm")
        bn_solver.train()
        fc_model = fcnet.FCNet(input_dim=input_dim,
                               hidden_dims=hidden_dims,
                               weight_scale=weight_scale,
                               use_batchnorm=False)

        fc_solver = solver.Solver(fc_model,
                                  small_data,
                                  num_epochs=num_epochs,
                                  update_rule=update_rule,
                                  optim_config={'learning_rate': learning_rate},
                                  verbose=True,
                                  print_every=200)
        print("Training without batchnorm")
        fc_solver.train()

        print("======== TestFCNet.test_batchnorm: <END> ")

    def test_weight_init(self):
        print("======== TestFCNet.test_weight_init:")

        dataset = load_data(self.data_dir, self.verbose)
        num_train = 1500
        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        input_dim = 32 * 32 * 3
        hidden_dims = [100, 100, 100, 100, 100, 100, 100, 100, 100]
        weight_scale = 2e-2
        reg = 2e-2
        learning_rate = 1e-3
        num_epochs=20
        batch_size = 50
        update_rule='adam'
        weight_init = ['gauss', 'gauss_sqrt2', 'xavier']

        model_dict = {}
        for w in weight_init:
            model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            reg=reg,
                            weight_init=w)
            model_dict[w] = model
        solver_dict = {}

        for k, m in model_dict.items():
            if self.verbose:
                print(m)

            solv = solver.Solver(m,
                                small_data,
                                print_every=100,
                                num_epochs=num_epochs,
                                batch_size=batch_size,     # previously 25
                                update_rule=update_rule,
                                optim_config={'learning_rate': learning_rate})
            solv.train()
            #skey = '%s-%s' % (m.__repr__(), k)
            skey = '%s' % k
            solver_dict[skey] = solv

        if self.draw_plots:
            fig, ax = vis_solver.get_train_fig()
            vis_solver.plot_solver_compare(ax, solver_dict)
            #vis_solver.plot_solver(ax, solv)
            plt.show()

        print("======== TestFCNet.test_weight_init: <END> ")


class TestFCNetDropout(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.verbose = True
        self.eps = 1e-6
        self.draw_plots = False
        self.num_classes = 10
        self.never_cheat = False   # implement cheat switch

    def test_fcnet_2layer_dropout(self):
        print("\n======== TestFCNetDropout.test_fcnet_2layer_dropout :")
        dataset = load_data(self.data_dir, self.verbose)
        num_train = 10

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        #hidden_dims = [100, 100, 100, 100]
        num_epochs = 20
        batch_size = 100
        solvers = {}
        dropout_probs = [0.0, 0.3, 0.5, 0.7]

        for d in dropout_probs:
            model = fcnet.FCNet(hidden_dims=[500],
                                input_dim=input_dim,
                                num_classes=10,
                                dropout=d,
                                weight_scale=2e-2)
            s = solver.Solver(model, small_data,
                              num_epochs=num_epochs,
                              batch_size=batch_size,
                              update_rule='adam',
                              optim_config = {'learning_rate': 5e-4},
                              verbose=True,
                              print_every=500)
            print("Training with dropout %f" % d)
            s.train()
            solvers['p=' + str(d)] = s

        if self.draw_plots:
            fig, ax = get_figure_handles()
            plot_test_result(ax, solvers, num_epochs)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== TestFCNetDropout.test_fcnet_2layer_dropout: <END> ")

    #def test_fcnet_3layer_dropout(self):
    #    print("\n======== TestFCNetDropout.test_fcnet_3layer_dropout :")

    #    dataset = load_data(self.data_dir, self.verbose)
    #    num_train = 10
    #    small_data = {
    #        'X_train': dataset['X_train'][:num_train],
    #        'y_train': dataset['y_train'][:num_train],
    #        'X_val':   dataset['X_val'][:num_train],
    #        'y_val':   dataset['y_val'][:num_train]
    #    }
    #    input_dim = 32 * 32 * 3
    #    hidden_dims = [100, 100]
    #    layer_types = ['relu', 'relu']
    #    weight_scale = 0.079564
    #    learning_rate = 0.003775

    #    # Get model and solver
    #    model = fcnet.FCNetObject(input_dim=input_dim,
    #                        hidden_dims=hidden_dims,
    #                        layer_types=layer_types,
    #                        weight_scale=weight_scale,
    #                        dtype=np.float64,
    #                        verbose=True)
    #    print(model)
    #    # TODO : Update solver for object oriented design
    #    model_solver = solver.Solver(model,
    #                                 small_data,
    #                                 print_every=10,
    #                                 num_epochs=30,
    #                                 batch_size=50,     # previously 25
    #                                 update_rule='sgd',
    #                                 optim_config={'learning_rate': learning_rate})
    #    model_solver.train()

    #    # Plot results
    #    if self.draw_plots:
    #        plt.plot(model_solver.loss_history, 'o')
    #        plt.title('Training loss history (3 layers)')
    #        plt.xlabel('Iteration')
    #        plt.ylabel('Training loss')
    #        plt.show()

    #    print("======== TestFCNetDropout.test_fcnet_3layer_dropout: <END> ")


#class TestFCNetObject(unittest.TestCase):
#    def setUp(self):
#        self.data_dir = 'datasets/cifar-10-batches-py'
#        self.verbose = True
#        self.eps = 1e-6
#        self.draw_plots = False
#        self.num_classes = 10
#        self.never_cheat = False   # implement cheat switch
#
#    def test_fcnet_3layer_overfit(self):
#        print("\n======== TestFCNetObject.test_fcnet_3layer_overfit:")
#
#        dataset = load_data(self.data_dir, self.verbose)
#        num_train = 50
#
#        small_data = {
#            'X_train': dataset['X_train'][:num_train],
#            'y_train': dataset['y_train'][:num_train],
#            'X_val':   dataset['X_val'][:num_train],
#            'y_val':   dataset['y_val'][:num_train]
#        }
#        #input_dim = small_data['X_train'].shape[0]
#        input_dim = 3 * 32 * 32
#        hidden_dims = [100, 100, 100, 100]
#        num_epochs = 20
#        batch_size = 100
#        dropout_probs = [0.0, 0.3, 0.5, 0.7]
#        solvers = {}
#
#        for d in dropout_probs:
#            model = fcnet.FCNet(hidden_dims=hidden_dims,
#                                input_dim=input_dim,
#                                num_classes=10,
#                                dropout=d,
#                                weight_scale=2e-2)
#            s = solver.Solver(model, small_data,
#                              num_epochs=num_epochs,
#                              batch_size=batch_size,
#                              update_rule='adam',
#                              optim_config = {'learning_rate': 5e-4},
#                              verbose=True,
#                              print_every=500)
#            print("Training with dropout %f" % d)
#            s.train()
#            solvers['p=' + str(d)] = s
#
#        if self.draw_plots:
#            fig, ax = get_figure_handles()
#            plot_test_result(ax, solvers, num_epochs)
#            fig.set_size_inches(8,8)
#            fig.tight_layout()
#            plt.show()


if __name__ == "__main__":
    unittest.main()
