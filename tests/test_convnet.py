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
from pymllib.classifiers import convnet
import pymllib.solver.solver as solver
import pymllib.vis.vis_weights as vis_weights

# Debug
from pudb import set_trace; set_trace()

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

    assert len(ax) == 3, "Need 3 axis"
    #assert len(weight_dict.keys()) == 3, "Need 3 sets of weights"

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
        self.draw_plots = True

    def test_conv_forward_naive(self):
        print("\n======== TestConvNet.test_conv_forward_naive:")

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

        out, cache = layers.conv_forward_naive(X, W, b, conv_param)
        dx, dw, db = layers.conv_backward_naive(dout, cache)

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
        num_epochs = 50
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
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,
                                    update_rule=update_rule,
                                    optim_config={'learning_rate': learning_rate},
                                    print_every=10,
                                    verbose=self.verbose)
        conv_solver.train()
        conv_dict = {"convnet": conv_solver}
        # Plot figures
        if self.draw_plots is True:
            fig, ax = get_figure_handles()
            plot_test_result(ax, conv_dict, num_epochs)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== TestConvNet.test_overfit_3layer: <END> ")


        # TODO : Next up, spatial batch normalization



"""
All the old tests have been temporarily moved here
"""
class Test3LayerConvNet(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.eps = 1e-7
        self.verbose = True
        self.draw_plots = True

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

        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.conv_forward_naive(X, W, b, conv_param)[0], X, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(lambda w: layers.conv_forward_naive(X, W, b, conv_param)[0], W, dout)
        db_num = check_gradient.eval_numerical_gradient_array(lambda b: layers.conv_forward_naive(X, W, b, conv_param)[0], b, dout)

        out, cache = layers.conv_forward_naive(X, W, b, conv_param)
        dx, dw, db = layers.conv_backward_naive(dout, cache)

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


    def test_gradient_check_conv(self):
        print("\n======== Test3LayerConvNet.test_gradient_check_conv:")

        num_inputs = 2
        input_dim = (3, 10, 10)
        reg = 0.0
        num_classes = 10

        X = np.random.randn(num_inputs, *input_dim)
        y = np.random.randint(num_classes, size=num_inputs)

        model = convnet.ThreeLayerConvNet(num_filters=3,
                                          filter_size=3,
                                          input_dim=input_dim,
                                          hidden_dim=7,
                                          reg=reg,
                                          dtype=np.float32)
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

        print("======== Test3LayerConvNet.test_gradient_check_conv: <END> ")

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
        input_dim = (3, 32, 32)
        weight_scale = 0.07
        learning_rate = 0.007
        num_epochs = 20
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
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,
                                    update_rule=update_rule,
                                    optim_config={'learning_rate': learning_rate},
                                    print_every=10,
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

class TestConvImgProc(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'datasets/'

    def test_conv_filter(self):
        print("\n======== TestConvImgProc.test_conv_filter:")

        from scipy.misc import imread, imresize
        import matplotlib.pyplot as plt

        img_filenames = [str(self.data_dir) + 'kitten.jpg', str(self.data_dir) + 'puppy.jpg']
        kitten = imread(img_filenames[0])
        puppy = imread(img_filenames[1])
        # Manipulate dims
        d = kitten.shape[1] - kitten.shape[0]
        kitten_cropped = kitten[:, int(d/2) : int(-d/2), :]

        img_size = 200
        X = np.zeros((2, 3, img_size, img_size))            # Input data
        X[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
        X[1:,:, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

        # Set up convolutional weights holding 2 filters, each 3x3
        W = np.zeros((2, 3, 3, 3))
        # The first filter converts the image to grayscale
        # Set up red, green and blue channels of filter
        W[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
        W[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
        W[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
        # Second filter detects horizontal edges in the blue channel
        W[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        # Vector of biases. No biases are needed for grayscale filter but for
        # the edge detection filter we want to add 128 to each output so that
        # no results are negative
        b = np.array([0, 128])

        # Compute the result of convolving each input in X with each filter in
        # W
        out, _ = layers.conv_forward_naive(X, W, b, {'stride': 1, 'pad': 1})

        # Tiny helper for showing images as uint8's and
        # removing axis labels
        def imshow_noax(img, normalize=True):
            if normalize is True:
                img_max = np.max(img)
                img_min = np.min(img)
                img = 255.0 * (img - img_min) / (img_max - img_min)
            plt.imshow(img.astype('uint8'))
            plt.gca().axis('off')

        # Show the original images and the results of the conv operation
        plt.subplot(2, 3, 1)
        imshow_noax(puppy, normalize=False)
        plt.title('Original')
        plt.subplot(2, 3, 2)
        imshow_noax(out[0, 0])
        plt.title('Grayscale')
        plt.subplot(2, 3, 3)
        imshow_noax(out[0, 1])
        plt.title('Horizontal Edges')
        plt.subplot(2, 3, 4)
        imshow_noax(kitten_cropped, normalize=False)
        plt.title('Original')
        plt.subplot(2, 3, 5)
        imshow_noax(out[1, 0])
        plt.title('Grayscale')
        plt.subplot(2, 3, 6)
        imshow_noax(out[1, 1])
        plt.title('Horizontal Edges')
        plt.show()

        print("======== TestConvImgProc.test_conv_filter: <END> ")


if __name__ == "__main__":
    unittest.main()
