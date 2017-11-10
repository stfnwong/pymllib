"""
TEST_SOLVER
Test the solver object and the various optimization functions
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import matplotlib.pyplot as plt
import numpy as np
import pymllib.utils.error as error
import pymllib.utils.data_utils as data_utils
import pymllib.solver.solver as solver
import pymllib.solver.optim as optim
import pymllib.classifiers.fcnet as fcnet

# Debug
#from pudb import set_trace; set_trace()

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
        self.draw_fig = True
        self.verbose = False
        self.draw_plots = False

    # CS231n test
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

    # CS231n test
    def test_rmsprop(self):
        print("\n======== TestSolver.test_rmsprop:")

        N = 4
        D = 5
        w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
        cache = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
        config = {'learning_rate': 1e-2, 'cache': cache}
        next_w, _ = optim.rmsprop(w, dw, config=config)

        expected_next_w = np.asarray([
        [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
        [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
        [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
        [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
        expected_cache = np.asarray([
        [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
        [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
        [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
        [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])

        next_w_error = error.rel_error(next_w, expected_next_w)
        cache_error = error.rel_error(config['cache'], expected_cache)
        print("next_w_error = %f" % next_w_error)
        print("cache_error = %f" % cache_error)
        self.assertLessEqual(next_w_error, self.eps)
        self.assertLessEqual(cache_error, self.eps)

        print("======== TestSolver.test_rmsprop: <END> ")


    def test_adam(self):
        print("\n======== TestSolver.test_adam:")

        N = 4
        D = 5
        w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
        m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
        v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)
        config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}

        next_w, _ = optim.adam(w, dw, config=config)
        expected_next_w = np.asarray([
        [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
        [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929 ],
        [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
        [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
        expected_v = np.asarray([
        [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
        [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
        [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
        [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966    ]])
        expected_m = np.asarray([
        [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
        [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
        [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
        [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])

        next_w_error = error.rel_error(next_w, expected_next_w)
        v_error = error.rel_error(config['v'], expected_v)
        m_error = error.rel_error(config['m'], expected_m)

        print("next_w_error = %f" % next_w_error)
        print("v_error = %f" % v_error)
        print("m_error = %f" % m_error)

        self.assertLessEqual(next_w_error, self.eps)
        self.assertLessEqual(v_error, self.eps)
        self.assertLessEqual(m_error, self.eps)

        print("======== TestSolver.test_adam: <END> ")


class TestSolverFCNet(unittest.TestCase):

    def setUp(self):
        self.eps = 1e-6
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.draw_fig = True
        self.verbose = False

    def test_rmsprop_fcnet(self):
        print("\n======== TestSolverFCNet.test_rmsprop_fcnet:")
        dataset =  load_data(self.data_dir, self.verbose)
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
        update_rule = 'rmsprop'

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
        model_solver.train()

        if self.draw_fig is True:
            solvers = {'rmsprop': model_solver}
            fig, ax = get_figure_handles()
            plot_test_result(ax, solvers, num_epochs)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== TestSolverFCNet.test_rmsprop_fcnet: <END> ")

    # TODO : Perhaps change this to just adam
    def test_adam_vs_rmsprop_fcnet(self):
        print("\n======== TestSolverFCNet.test_adam_vs_rmsprop:")
        dataset =  load_data(self.data_dir, self.verbose)
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
        hidden_dims = [100, 100, 100, 100, 100]
        weight_scale = 5e-2
        num_epochs = 20
        batch_size = 50
        reg = 1e-1
        lr = {'rmsprop': 1e-4, 'adam': 1e-3}
        update_rule = ['rmsprop', 'adam']

        solvers = {}
        for u in update_rule:
            model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            reg=reg,
                            dtype=np.float64)
            if self.verbose:
                print(model)
            model_solver = solver.Solver(model,
                                        small_data,
                                        print_every=100,
                                        num_epochs=num_epochs,
                                        batch_size=batch_size,     # previously 25
                                        update_rule=u,
                                        optim_config={'learning_rate': lr[u]})
            solvers[u] = model_solver
            model_solver.train()

        if self.draw_fig is True:
            fig, ax = get_figure_handles()
            plot_test_result(ax, solvers, num_epochs)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()



        print("======== TestSolverFCNet.test_adam_vs_rmsprop: <END> ")

    def test_all_optim_fcnet_5layer(self):
        print("\n======== TestSolverFCNet.test_all_optim_fcnet_5layer:")

        dataset =  load_data(self.data_dir, self.verbose)
        num_train = 50
        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['y_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['y_val'][:num_train]
        }
        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        hidden_dims = [100, 100, 100, 100, 100]
        #hidden_dims = [100, 50, 10]     # just some random dims
        weight_scale = 5e-2
        reg = 1e-1
        num_epochs = 30
        batch_size = 50
        solvers = {}

        # Solver params
        optim_list = ['rmsprop', 'sgd_momentum', 'adam', 'sgd']
        lr = {'rmsprop': 1e-4, 'adam': 1e-3, 'sgd': 1e-3, 'sgd_momentum': 1e-3}

        for update_rule in optim_list:
            print("Using update rule %s" % update_rule)
            model = fcnet.FCNet(input_dim=input_dim,
                            hidden_dims=hidden_dims,
                            weight_scale=weight_scale,
                            reg=reg,
                            dtype=np.float64)
            if self.verbose:
                print(model)
            model_solver = solver.Solver(model,
                                        small_data,
                                        print_every=100,
                                        num_epochs=num_epochs,
                                        batch_size=batch_size,     # previously 25
                                        update_rule=update_rule,
                                        optim_config={'learning_rate': lr[update_rule]})
            solvers[update_rule] = model_solver
            model_solver.train()

        # get some figure handles and plot the data
        if self.draw_plots:
            fig, ax = get_figure_handles()
            plot_test_result(ax, solvers, num_epochs)
            fig.set_size_inches(8,8)
            fig.tight_layout()
            plt.show()

        print("======== TestSolverFCNet.test_all_optim_fcnet_5layer: <END> ")


class TestSolverCheckpoint(unittest.TestCase):
    """
    Tests that a model can be written to and read from disk
    """

    def setUp(self):
        self.eps = 1e-6
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.draw_fig = True
        self.verbose = False

    def test_model_restore(self):
        print("\n======== TestSolverCheckpoint.test_model_restore:")

        dataset =  load_data(self.data_dir, self.verbose)
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
        update_rule = 'adam'

        model = fcnet.FCNet(input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        weight_scale=weight_scale,
                        dtype=np.float64)
        if self.verbose:
            print(model)
        ref_solver = solver.Solver(model,
                                    small_data,
                                    print_every=100,
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,     # previously 25
                                    update_rule=update_rule,
                                    optim_config={'learning_rate': learning_rate})
        ref_solver.train()
        ref_solver_file = 'tests/ref_solver.pkl'
        ref_solver.save(ref_solver_file)

        test_solver = solver.Solver(model, small_data)
        test_solver.load(ref_solver_file)

        # Compare the two solvers
        print("Checking parameters")
        self.assertEqual(ref_solver.update_rule, test_solver.update_rule)
        self.assertEqual(ref_solver.num_epochs, test_solver.num_epochs)
        self.assertEqual(ref_solver.batch_size, test_solver.batch_size)
        self.assertEqual(ref_solver.print_every, test_solver.print_every)
        self.assertEqual(ref_solver.lr_decay, test_solver.lr_decay)

        #self.assertDictEqual(ref_solver.optim_config, test_solver.optim_config)

        print("======== TestSolverCheckpoint.test_model_restore: <END> ")


if __name__ == "__main__":
    unittest.main()
