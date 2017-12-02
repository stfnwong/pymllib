"""
VIS_WEIGHTS
Various ways to visualize the weights in a neural network.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pymllib.solver.solver as solver
import pymllib.utils.convnet_utils as cutils

# Debug
#from pudb import set_trace; set_trace()

def vis_grid_tensor(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data into a grid for easy
    visualization.

    Inputs:
        - Xs: Data of shape (N, H, W, C)
        - ubound : Output grid will have values scaled to the range [0, ubound]
        - padding : The number of blank pixels between elements of the grid.
    """

    (N, H, W, C) = Xs.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))

    next_idx = 0
    y0 = 0
    y1 = H
    for y in range(grid_size):
        x0 = 0
        x1 = W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low = np.min(img)
                high = np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding

    return grid


def vis_grid_img(Xs):
    """
    Visualize a grid of images
    """
    (N, H, W, C) = Xs.shape
    A = int(np.ceil(np.sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)

    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y:(y+1) * H + y, x * W + x:(x+1)*W+x, :] = Xs[n, :, :, :]
                n += 1
    # Normalize
    gmax = np.max(G)
    gmin = np.min(G)
    G = (G - gmin) / (gmax - gmin)

    return G


def vis_nn(rows):
    """
    Visualize an array of arrays of images
    """

    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)

    for y in range(N):
        for x in range(D):
            G[y * H + y:(y+1) ** H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # Normalize
    gmin = np.min(G)
    gmax = np.max(G)
    G = (G - gmin) / (gmax - gmin)

    return G


def inspect_checkpoint(checkpoint_fname, verbose=False):

    csolver = solver.Solver(None, None)
    # TODO : Check file exists
    csolver.load(checkpoint_fname)
    weight_dict = cutils.get_conv_layer_dict(csolver.model)

    # For now, get the first layer weights out and show those
    grid = vis_grid_img(weight_dict['W1'].transpose(0, 2, 3, 1))

    def get_fig_handle():
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        return fig, ax

    fig, ax = get_fig_handle()
    ax.imshow(grid)




if __name__ == "__main__":
    print("TODO: Entry point for test")
