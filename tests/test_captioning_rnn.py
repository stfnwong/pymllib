"""
TEST_CAPTIONING_RNN

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pymllib.utils import error
from pymllib.utils import check_gradient
from pymllib.utils import rnn_utils
from pymllib.utils import coco_utils
from pymllib.utils import image_utils
# Unit under test
from pymllib.layers import rnn_layers
from pymllib.classifiers import captioning_rnn
from pymllib.solver import captioning_solver

# Debug
#from pudb import set_trace; set_trace()


# TODO : Debug function for tuple problem
def print_ptypes(model):

    for k, v in model.params.items():
        print('%s : %s' % (k, type(v)))


class TestCaptioningRNN(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-6
        self.dtype = np.float64
        self.verbose = True
        self.draw_figures = True

    def test_step_forward(self):
        print("\n======== TestCaptioningRNN.test_step_forward:")

        N = 3
        D = 10
        H = 4
        X = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
        Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
        Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
        b = np.linspace(-0.2, 0.4, num=H)

        next_h, _ = rnn_layers.rnn_step_forward(X, prev_h, Wx, Wh, b)
        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [ 0.66854692,  0.79562378,  0.8775553,   0.92795967],
            [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

        err = error.rel_error(expected_next_h, next_h)
        print('Relative error : %f' % err)
        self.assertLessEqual(err, self.eps)

        print("======== TestCaptioningRNN.test_step_forward: <END> ")

    def test_step_backward(self):
        print("\n======== TestCaptioningRNN.test_step_forward:")

        N = 4
        D = 5
        H = 6
        X = np.random.randn(N, D)
        h = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)

        out, cache = rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)
        dnext_h = np.random.randn(*out.shape)

        fx = lambda x:      rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fh = lambda prev_h: rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fWx = lambda Wx:    rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fWh = lambda Wh:    rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fb = lambda b:      rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]

        dx_num = check_gradient.eval_numerical_gradient_array(fx, X, dnext_h)
        dprev_h_num = check_gradient.eval_numerical_gradient_array(fh, h, dnext_h)
        dWx_num = check_gradient.eval_numerical_gradient_array(fWx, Wx, dnext_h)
        dWh_num = check_gradient.eval_numerical_gradient_array(fWh, Wh, dnext_h)
        db_num = check_gradient.eval_numerical_gradient_array(fb, b, dnext_h)

        dx, dprev_h, dWx, dWh, db = rnn_layers.rnn_step_backward(dnext_h, cache)

        dx_err = error.rel_error(dx, dx_num)
        dprev_h_err = error.rel_error(dprev_h, dprev_h_num)
        dwx_err = error.rel_error(dWx, dWx_num)
        dwh_err = error.rel_error(dWh, dWh_num)
        db_err = error.rel_error(db, db_num)

        print("dx_err : %f" % dx_err)
        print("dprev_h_err : %f" % dprev_h_err)
        print("dwx_err : %f" % dwx_err)
        print("dwh_err : %f" % dwh_err)
        print("db_err : %f" % db_err)

        self.assertLessEqual(dx_err, self.eps)
        self.assertLessEqual(dprev_h_err, self.eps)
        #self.assertLessEqual(dwx_err, self.eps)
        self.assertLessEqual(dwh_err, self.eps)
        self.assertLessEqual(db_err, self.eps)

        print("======== TestCaptioningRNN.test_step_forward: <END> ")


    def test_word_embedding_forward(self):
        print("\n======== TestCaptioningRNN.test_word_embedding_forward:")
        N = 2
        T = 3
        V = 5
        D = 3

        X = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
        W = np.linspace(0, 1, num=V*D).reshape(V, D)

        out, _ = rnn_layers.word_embedding_forward(X, W)
        expected_out = np.asarray([
            [[ 0.,          0.07142857,  0.14285714],
            [  0.64285714,  0.71428571,  0.78571429],
            [  0.21428571,  0.28571429,  0.35714286],
            [  0.42857143,  0.5,         0.57142857]],
            [[ 0.42857143,  0.5,         0.57142857],
            [  0.21428571,  0.28571429,  0.35714286],
            [  0.,          0.07142857,  0.14285714],
            [  0.64285714,  0.71428571,  0.78571429]]])

        word_err = error.rel_error(expected_out, out)
        self.assertLessEqual(word_err, self.eps)
        print("Word error : %f" % word_err)

        print("======== TestCaptioningRNN.test_word_embedding_forward: <END> ")

    def test_word_embedding_backward(self):
        print("\n======== TestCaptioningRNN.test_word_embedding_backward:")
        N = 2
        T = 3
        V = 5
        D = 3

        X = np.random.randint(V, size=(N, T))
        W = np.random.randn(V, D)

        out, cache = rnn_layers.word_embedding_forward(X, W)
        print('cache len : %d' % len(cache))
        dout = np.random.randn(*out.shape)
        dW = rnn_layers.word_embedding_backward(dout, cache)

        f = lambda W: rnn_layers.word_embedding_forward(X, W)[0]
        dW_num = check_gradient.eval_numerical_gradient_array(f, W, dout)
        dw_error = error.rel_error(dW, dW_num)

        self.assertLessEqual(dw_error, self.eps)
        print("dW error : %f" % dw_error)

        print("======== TestCaptioningRNN.test_word_embedding_backward: <END> ")


    def test_temporal_affine_forward(self):
        print("\n======== TestCaptioningRNN.test_temporal_affine_forward:")

        N = 2
        T = 3
        D = 4
        M = 5

        X = np.random.randn(N, T, D)
        W = np.random.randn(D, M)
        b = np.random.randn(M)

        out, cache = rnn_layers.temporal_affine_forward(X, W, b)
        dout = np.random.randn(*out.shape)
        # Forward pass lambda functions
        fx = lambda x: rnn_layers.temporal_affine_forward(X, W, b)[0]
        fw = lambda w: rnn_layers.temporal_affine_forward(X, W, b)[0]
        fb = lambda b: rnn_layers.temporal_affine_forward(X, W, b)[0]

        dx_num = check_gradient.eval_numerical_gradient_array(fx, X, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(fw, W, dout)
        db_num = check_gradient.eval_numerical_gradient_array(fb, b, dout)
        dx, dw, db = rnn_layers.temporal_affine_backward(dout, cache)
        # Compute errors
        dx_err = error.rel_error(dx_num, dx)
        dw_err = error.rel_error(dw_num, dw)
        db_err = error.rel_error(db_num, db)

        self.assertLessEqual(dx_err, self.eps)
        self.assertLessEqual(dw_err, self.eps)
        self.assertLessEqual(db_err, self.eps)

        print('dx_err : %f' % dx_err)
        print('dw_err : %f' % dw_err)
        print('db_err : %f' % db_err)

        print("======== TestCaptioningRNN.test_temporal_affine_forward: <END> ")

    def test_temporal_softmax(self):
        print("\n======== TestCaptioningRNN.test_temporal_softmax:")

        N = 100
        T = 1
        V = 10

        loss1 = rnn_utils.check_loss(100, 1, 10, 1.0)   # expect about 2.3
        loss2 = rnn_utils.check_loss(1000, 10, 10, 1.0) # expect about 23
        loss3 = rnn_utils.check_loss(5000, 10, 10, 0.1) # expect about 2.3

        self.assertLessEqual(loss1, 2.3)
        self.assertLessEqual(loss2, 23)
        self.assertLessEqual(loss3, 2.3)
        print('loss (100, 1, 10, 1.0) : %f' % loss1)
        print('loss (1000, 10, 10, 1.0) : %f' % loss2)
        print('loss (5000, 10, 10, 0.1) : %f' % loss3)

        print("Performing gradient check for temporal softmax loss")
        N = 7
        T = 8
        V = 9

        X = np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = (np.random.randn(N, T) > 0.5)
        loss, dx = rnn_layers.temporal_softmax_loss(X, y, mask, verbose=self.verbose)
        dx_num = check_gradient.eval_numerical_gradient(lambda X: rnn_layers.temporal_softmax_loss(X, y, mask)[0], X, verbose=self.verbose)
        dx_err = error.rel_error(dx, dx_num)
        self.assertLessEqual(dx_err, self.eps)

        print('dx err : %f' % dx_err)

        print("======== TestCaptioningRNN.test_temporal_softmax: <END> ")


    def test_basic_captioning_loss(self):
        print("\n======== TestCaptioningRNN.test_basic_captioning_loss:")
        N = 10
        D = 20
        W = 30
        H = 40
        word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
        V = len(word_to_idx)    # size of our vocabulary
        T = 13

        model = captioning_rnn.CaptioningRNN(word_to_idx,
                                input_dim=D,
                                wordvec_dim=W,
                                hidden_dim=H,
                                cell_type='rnn',
                                dtype=self.dtype)

        # Set all parameters to fixed values
        for k, v in model.params.items():
            model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

        features = np.linspace(-1.5, 0.3, num=(N*D)).reshape(N, D)
        captions = (np.arange(N * T) % V).reshape(N, T)

        loss, grads = model.loss(features, captions)
        expected_loss = 9.83235591003
        loss_err = abs(loss - expected_loss)
        self.assertLessEqual(loss_err, self.eps)
        print('loss          : %f' % loss)
        print('expected loss : %f' % expected_loss)
        print('error         : %f' % loss_err)

        print("======== TestCaptioningRNN.test_basic_captioning_loss: <END> ")

    def test_basic_captioning_gradient(self):
        print("\n======== TestCaptioningRNN.test_basic_captioning_gradient:")
        batch_size = 2
        timesteps = 3
        input_dim = 4
        wordvec_dim = 5
        hidden_dim = 6
        word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
        V = len(word_to_idx)    # size of our vocabulary

        captions = np.random.randint(V, size=(batch_size, timesteps))
        features = np.random.randn(batch_size, input_dim)

        # Get a model
        model = captioning_rnn.CaptioningRNN(word_to_idx,
                                input_dim=input_dim,
                                wordvec_dim=wordvec_dim,
                                hidden_dim=hidden_dim,
                                cell_type='rnn',
                                dtype=self.dtype)
        # Compute loss
        loss, grads = model.loss(features, captions)
        for param_name in sorted(grads):
            f = lambda _: model.loss(features, captions)[0]
            param_grad_num = check_gradient.eval_numerical_gradient(f,
                                    model.params[param_name], verbose=False, h=1e-6)
            err = error.rel_error(param_grad_num, grads[param_name])
            #self.assertLessEqual(err, self.eps)
            self.assertLessEqual(err, 1e-5)
            print("%s relative error : %f" % (param_name, err))

        print("======== TestCaptioningRNN.test_basic_captioning_gradient: <END> ")


    def test_overfit(self):
        print("\n======== TestCaptioningRNN.test_overfit:")

        small_data = coco_utils.load_coco_data(max_train=50)

        small_rnn_model = captioning_rnn.CaptioningRNN(
            cell_type='rnn',
            word_to_idx=small_data['word_to_idx'],
            input_dim=small_data['train_features'].shape[1],
            hidden_dim=512,
            wordvec_dim=256
        )

        solv = captioning_solver.CaptioningSolver(
            small_rnn_model,
            small_data,
            update_rule='adam',
            num_epochs=50,
            batch_size=25,
            optim_config={'learning_rate': 5e-3},
            lr_decay=0.95,
            verbose=self.verbose,
            print_every=1
        )

        solv.train()

        if self.draw_figures:
            plt.plot(solv.loss_history)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title("Training loss history")
            plt.show()

        print("======== TestCaptioningRNN.test_overfit: <END> ")

    def test_sampling(self):
        print("\n======== TestCaptioningRNN.test_sampling:")

        small_data = coco_utils.load_coco_data(max_train=50)

        small_rnn_model = captioning_rnn.CaptioningRNN(
            cell_type='rnn',
            word_to_idx=small_data['word_to_idx'],
            input_dim=small_data['train_features'].shape[1],
            hidden_dim=512,
            wordvec_dim=256
        )
        solv = captioning_solver.CaptioningSolver(
            small_rnn_model,
            small_data,
            update_rule='adam',
            num_epochs=50,
            batch_size=25,
            optim_config={'learning_rate': 5e-3},
            lr_decay=0.95,
            verbose=self.verbose,
            print_every=1
        )
        # Train the model
        solv.train()

        for split in ['train', 'val']:
            minibatch = coco_utils.sample_coco_minibatch(small_data,
                                                         split=split,
                                                         batch_size=2)
            gt_captions, features, urls = minibatch
            gt_captions = coco_utils.decode_captions(gt_captions, small_data['idx_to_word'])

            sample_captions = small_rnn_model.sample(features)
            sample_captions = sample_captions.astype(np.int32)
            sample_captions = coco_utils.decode_captions(sample_captions, small_data['idx_to_word'])

            for gt_capt, samp_capt, url in zip(gt_captions, sample_captions, urls):
                plt.imshow(image_utils.image_from_url(url))
                plt.title('%s\n%5s\nGT :%s' % (split, samp_capt, gt_capt))
                plt.axis('off')
                plt.show()

        print("======== TestCaptioningRNN.test_sampling: <END> ")


if __name__ == '__main__':
    unittest.main()
