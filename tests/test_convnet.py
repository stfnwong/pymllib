"""
TEST_CONVNET
Test the convolutional network classifier layers. This is essentially just
a re-factor of the CS231n exercise
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pymllib.utils import data_utils
from pymllib.utils import check_gradient
from pymllib.utils import error
from pymllib.layers import layers
from pymllib.layers import conv_layers
from pymllib.classifiers import convnet
from pymllib.solver import solver
from pymllib.vis import vis_weights
from pymllib.vis import vis_solver

# Debug
#from pudb import set_trace; set_trace()

# TODO : Once an arbitrary layer convnet is implemented,
# replace the three layer convnet with the arbitrary convnet
# implemented with 3 layers

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

def plot_3layer_activations(ax, weight_dict):
    if len(ax) != 3:
        raise ValueError('Need 3 axis for this method')

    for n in range(len(ax)):
        grid = vis_weights.vis_grid_img(weight_dict['W' + str(n+1)].transpose(0, 2, 3, 1))
        ax[n].imshow(grid.astype('uint8'))
        title = "W%d" % n+1
        ax[n].set_title(title)


class TestConvNet(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.eps = 1e-7
        self.verbose = True
        self.draw_plots = False
        self.print_every = 500
        self.num_epochs = 10

    def test_conv_forward_naive(self):
        print("\n======== TestConvNet.test_conv_forward_naive:")

        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = conv_layers.conv_forward_naive(x, w, b, conv_param)
        correct_out = np.array([[[[[-0.08759809, -0.10987781],
                                   [-0.18387192, -0.2109216 ]],
                                  [[ 0.21027089,  0.21661097],
                                   [ 0.22847626,  0.23004637]],
                                  [[ 0.50813986,  0.54309974],
                                   [ 0.64082444,  0.67101435]]],
                                 [[[-0.98053589, -1.03143541],
                                   [-1.19128892, -1.24695841]],
                                  [[ 0.69108355,  0.66880383],
                                   [ 0.59480972,  0.56776003]],
                                  [[ 2.36270298,  2.36904306],
                                   [ 2.38090835,  2.38247847]]]]])
        out_error = error.rel_error(out, correct_out)
        print("out_error : %.9f " % out_error)
        self.assertLessEqual(out_error, self.eps)

        print("======== TestConvNet.test_conv_forward_naive: <END> ")

    def test_conv_backward_naive(self):
        print("\n======== TestConvNet.test_conv_backward_naive:")
        X = np.random.randn(4, 3, 5, 5)
        W = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2,)
        dout = np.random.randn(4, 2, 5, 5)
        conv_param = {'stride': 1, 'pad': 1}

        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.conv_forward_naive(X, W, b, conv_param)[0], X, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(lambda w: layers.conv_forward_naive(X, W, b, conv_param)[0], W, dout)
        db_num = check_gradient.eval_numerical_gradient_array(lambda b: layers.conv_forward_naive(X, W, b, conv_param)[0], b, dout)

        out, cache = conv_layers.conv_forward_naive(X, W, b, conv_param)
        dx, dw, db = conv_layers.conv_backward_naive(dout, cache)

        dx_error = error.rel_error(dx, dx_num)
        dw_error = error.rel_error(dw, dw_num)
        db_error = error.rel_error(db, db_num)

        print("dx_error : %.9f" % dx_error)
        print("dw_error : %.9f" % dw_error)
        print("db_error : %.9f" % db_error)

        self.assertLessEqual(dx_error, self.eps)
        self.assertLessEqual(dw_error, self.eps)
        self.assertLessEqual(db_error, self.eps)

        print("======== TestConvNet.test_conv_backward_naive: <END> ")

    def test_loss_2conv_layers(self):

        print("\n======== TestConvNet.test_loss_3layer_conv:")

        N = 10       # Because the naive implementation is VERY slow
        X = np.random.randn(N, 3, 32, 32)
        y = np.random.randint(10, size=N)
        model_3l = convnet.ConvNetLayer()
        model_3l.reg = 0.0
        loss, grads = model_3l.loss(X,y)
        print("Initial loss (no regularization) : %f" % loss)
        model_3l.reg = 0.5
        loss, grads = model_3l.loss(X, y)
        print("Initial loss (with regularization) : %f" % loss)

        print("======== TestConvNet.test_loss_3layer_conv: <END> ")


    def test_gradient_check_2conv_layers(self):
        print("\n======== TestConvNet.test_gradient_check_conv:")

        num_inputs = 2
        input_dim = (3, 32, 32)
        num_classes = 10

        X = np.random.randn(num_inputs, *input_dim)
        y = np.random.randint(num_classes, size=num_inputs)

        # TODO ; Modify this to be L Layer net
        model = convnet.ConvNetLayer(reg=0.0)
        loss, grads = model.loss(X, y)
        for p in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            param_grad_num = check_gradient.eval_numerical_gradient(f, model.params[p], verbose=False, h=1e-6)
            err = error.rel_error(param_grad_num, grads[p])
            print("%s max relative error: %e" % (p, err))

        # This is in a separate pass so that we can see all errors
        # printed to console before we invoke the assertions
        for p in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            param_grad_num = check_gradient.eval_numerical_gradient(f, model.params[p], verbose=False, h=1e-6)
            err = error.rel_error(param_grad_num, grads[p])
            self.assertLessEqual(err, self.eps)

        print("======== TestConvNet.test_gradient_check_conv: <END> ")

    def test_overfit_3layer(self):
        print("\n======== TestConvNet.test_overfit_3layer:")
        dataset = load_data(self.data_dir, self.verbose)
        num_train = 500

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        if self.verbose:
            print("Size of training dataset :")
            for k, v in small_data.items():
                print("%s : %s " % (k, v.shape))

        #weight_scale = 1e-2
        #learning_rate = 1e-3
        weight_scale = 0.06
        learning_rate = 0.077
        batch_size = 50
        update_rule='adam'

        # Get a model
        model = convnet.ConvNetLayer(weight_scale=weight_scale,
                                     num_filters=[32],
                                     hidden_dims=[100],
                                     use_batchnorm=True,
                                     reg=0.0)
        if self.verbose:
            print(model)
        # Get a solver
        conv_solver = solver.Solver(model,
                                    small_data,
                                    num_epochs=self.num_epochs,
                                    batch_size=batch_size,
                                    update_rule=update_rule,
                                    optim_config={'learning_rate': learning_rate},
                                    print_every=self.print_every,
                                    verbose=self.verbose)
        conv_solver.train()
        conv_dict = {"convnet": conv_solver}
        # Plot figures
        if self.draw_plots is True:
            fig, ax = get_figure_handles()
            plot_test_result(ax, conv_dict, self.num_epochs)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== TestConvNet.test_overfit_3layer: <END> ")


        # TODO : Next up, spatial batch normalization

    #def test_xavier_overfit(self):
    #    print("\n======== TestConvNet.test_xavier_overfit:")
    #    dataset = load_data(self.data_dir, self.verbose)
    #    num_train = 1500

    #    small_data = {
    #        'X_train': dataset['X_train'][:num_train],
    #        'y_train': dataset['y_train'][:num_train],
    #        'X_val':   dataset['X_val'][:num_train],
    #        'y_val':   dataset['y_val'][:num_train]
    #    }
    #    input_dim = (3, 32, 32)
    #    hidden_dims = [256, 256]
    #    num_filters = [16, 32]
    #    #weight_scale = 0.07
    #    #learning_rate = 0.007
    #    weight_scale = 1e-3
    #    learning_rate = 1e-3
    #    batch_size = 50
    #    update_rule='adam'

    #    weight_init = ['gauss', 'gauss_sqrt', 'xavier']
    #    model_dict = {}

    #    for w in weight_init:
    #        model = convnet.ConvNetLayer(input_dim=input_dim,
    #                        hidden_dims=hidden_dims,
    #                        num_filters = num_filters,
    #                        weight_scale=weight_scale,
    #                        user_xavier=True,
    #                        verbose=True,
    #                        dtype=np.float32)
    #        model_dict[w] = model

    #    solver_dict = {}

    #    for k, m in model_dict.items():
    #        if self.verbose:
    #            print(m)
    #        solv = solver.Solver(m,
    #                             small_data,
    #                             print_every=self.print_every,
    #                             num_epochs=self.num_epochs,
    #                             batch_size=batch_size,
    #                             update_rule=update_rule,
    #                             optim_config={'learning_rate': learning_rate})
    #        solv.train()
    #        fname = '%s-solver-%d-epochs.pkl' % (k, int(self.num_epochs))
    #        solv.save(fname)
    #        skey = '%s-%s' % (m.__repr__(), k)
    #        solver_dict[skey] = solv

    #    # Plot results
    #    if self.draw_plots is True:
    #        fig, ax = vis_solver.get_train_fig()
    #        vis_solver.plot_solver_compare(ax, solver_dict)
    #        plt.show()

    #    print("======== TestConvNet.test_xavier_overfit: <END> ")

"""
All the old tests have been temporarily moved here
"""
class Test3LayerConvNet(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.eps = 1e-7
        self.verbose = True
        self.draw_plots = True
        self.print_every = 500

    def test_conv_forward_naive(self):
        print("\n======== Test3LayerConvNet.test_conv_forward_naive:")

        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = layers.conv_forward_naive(x, w, b, conv_param)
        correct_out = np.array([[[[[-0.08759809, -0.10987781],
                                   [-0.18387192, -0.2109216 ]],
                                  [[ 0.21027089,  0.21661097],
                                   [ 0.22847626,  0.23004637]],
                                  [[ 0.50813986,  0.54309974],
                                   [ 0.64082444,  0.67101435]]],
                                 [[[-0.98053589, -1.03143541],
                                   [-1.19128892, -1.24695841]],
                                  [[ 0.69108355,  0.66880383],
                                   [ 0.59480972,  0.56776003]],
                                  [[ 2.36270298,  2.36904306],
                                   [ 2.38090835,  2.38247847]]]]])
        out_error = error.rel_error(out, correct_out)
        print("out_error : %.9f " % out_error)
        self.assertLessEqual(out_error, self.eps)

        print("======== Test3LayerConvNet.test_conv_forward_naive: <END> ")

    def test_conv_backward_naive(self):
        print("\n======== Test3LayerConvNet.test_conv_backward_naive:")
        X = np.random.randn(4, 3, 5, 5)
        W = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2,)
        dout = np.random.randn(4, 2, 5, 5)
        conv_param = {'stride': 1, 'pad': 1}

        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: conv_layers.conv_forward_naive(X, W, b, conv_param)[0], X, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(lambda w: conv_layers.conv_forward_naive(X, W, b, conv_param)[0], W, dout)
        db_num = check_gradient.eval_numerical_gradient_array(lambda b: conv_layers.conv_forward_naive(X, W, b, conv_param)[0], b, dout)

        out, cache = conv_layers.conv_forward_naive(X, W, b, conv_param)
        dx, dw, db = conv_layers.conv_backward_naive(dout, cache)

        dx_error = error.rel_error(dx, dx_num)
        dw_error = error.rel_error(dw, dw_num)
        db_error = error.rel_error(db, db_num)

        print("dx_error : %.9f" % dx_error)
        print("dw_error : %.9f" % dw_error)
        print("db_error : %.9f" % db_error)

        self.assertLessEqual(dx_error, self.eps)
        self.assertLessEqual(dw_error, self.eps)
        self.assertLessEqual(db_error, self.eps)

        print("======== Test3LayerConvNet.test_conv_backward_naive: <END> ")

    def test_loss_3layer_conv(self):

        print("\n======== Test3LayerConvNet.test_loss_3layer_conv:")

        N = 10       # Because the naive implementation is VERY slow
        X = np.random.randn(N, 3, 32, 32)
        y = np.random.randint(10, size=N)
        model_3l = convnet.ThreeLayerConvNet()
        model_3l.reg = 0.0
        loss, grads = model_3l.loss(X,y)
        print("Initial loss (no regularization) : %f" % loss)
        model_3l.reg = 0.5
        loss, grads = model_3l.loss(X, y)
        print("Initial loss (with regularization) : %f" % loss)

        print("======== Test3LayerConvNet.test_loss_3layer_conv: <END> ")


    #def test_gradient_check_conv(self):
    #    print("\n======== Test3LayerConvNet.test_gradient_check_conv:")

    #    num_inputs = 2
    #    input_dim = (3, 10, 10)
    #    reg = 0.0
    #    num_classes = 10

    #    X = np.random.randn(num_inputs, *input_dim)
    #    y = np.random.randint(num_classes, size=num_inputs)

    #    model = convnet.ThreeLayerConvNet(num_filters=3,
    #                                      filter_size=3,
    #                                      input_dim=input_dim,
    #                                      hidden_dim=7,
    #                                      reg=reg,
    #                                      dtype=np.float32)
    #    loss, grads = model.loss(X, y)
    #    for p in sorted(grads):
    #        f = lambda _: model.loss(X, y)[0]
    #        param_grad_num = check_gradient.eval_numerical_gradient(f, model.params[p], verbose=False, h=1e-6)
    #        err = error.rel_error(param_grad_num, grads[p])
    #        print("%s max relative error: %e" % (p, err))

    #    # This is in a separate pass so that we can see all errors
    #    # printed to console before we invoke the assertions
    #    for p in sorted(grads):
    #        f = lambda _: model.loss(X, y)[0]
    #        param_grad_num = check_gradient.eval_numerical_gradient(f, model.params[p], verbose=False, h=1e-6)
    #        err = error.rel_error(param_grad_num, grads[p])
    #        self.assertLessEqual(err, self.eps)

    #    print("======== Test3LayerConvNet.test_gradient_check_conv: <END> ")

    def test_overfit_3layer(self):
        print("\n======== Test3LayerConvNet.test_overfit_3layer:")
        dataset = load_data(self.data_dir, self.verbose)
        num_train = 500

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        weight_scale = 0.07
        learning_rate = 0.007
        batch_size = 50
        update_rule='adam'

        # Get a model
        model = convnet.ThreeLayerConvNet(weight_scale=weight_scale,
                                          reg=0.0)
        if self.verbose:
            print(model)
        # Get a solver
        conv_solver = solver.Solver(model,
                                    small_data,
                                    num_epochs=self.num_epochs,
                                    batch_size=batch_size,
                                    update_rule=update_rule,
                                    optim_config={'learning_rate': learning_rate},
                                    print_every=self.print_every,
                                    verbose=self.verbose)
        conv_solver.train()
        conv_dict = {"convnet": conv_solver}
        # Plot figures
        if self.draw_plots is True:
            fig, ax = get_figure_handles()
            plot_test_result(ax, conv_dict)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== Test3LayerConvNet.test_overfit_3layer: <END> ")


        # TODO : Next up, spatial batch normalization



if __name__ == "__main__":
    unittest.main()
