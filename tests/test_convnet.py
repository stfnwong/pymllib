"""
TEST_CONVNET
Test the convolutional network classifier layers. This is essentially just
a re-factor of the CS231n exercise
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import numpy as np
from pymllib.utils import data_utils
from pymllib.utils import check_gradient
from pymllib.utils import error
from pymllib.layers import layers
#from pymllib.solver import solver
from pymllib.classifiers import convnet

# ==== OLD IMPORTS
#import numpy as np
#import pymllib.utils.data_utils as data_utils
#import pymllib.utils.check_gradient as check_gradient
#import pymllib.utils.error as error
#import pymllib.layers as layers
#import pymllib.solver as solver
#import pymllib.convnet as convnet

import unittest
# Debug
#from pudb import set_trace; set_trace()

def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

class TestConvNet(unittest.TestCase):

    def setUp(self):
        self.eps = 1e-7

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

    def test_loss_3layer_conv(self):

        print("\n======== TestConvNet.test_loss_3layer_conv:")

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

        print("======== TestConvNet.test_loss_3layer_conv: <END> ")


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
