"""
This is the RBF classifier from Peter Roelants github page
http://peterroelants.github.io/posts/neural_network_implementation_part03/

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Debug
#from pudb import set_trace; set_trace()

# Activation function for neuron
def rbf(z):
    return np.exp(-z**2)


# ==== FORWARD PASS ==== #
# Functions for computing forward step
def logistic(z):
    return 1 / (1 + np.exp(-z))

def hidden_activations(x, wh):
    return rbf(x * wh)

def output_activations(h, wo):
    return logistic(h * wo - 1)

# network function
def nn(x, wh, wo):
    return output_activations(hidden_activations(x, wh), wo)

# The prediction function - this returns the predicted class score
def nn_predict(x, wh, wo):
    return np.around(nn(x, wh, wo))


# ==== BACKWARD PASS ==== #
def cost(y, t):
    # TODO : Split this one-liner into multiple lines
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

def cost_for_param(x, wh, wo, t):
    return cost(nn(x, wh, wo), t)

# Gradients functions
def gradient_output(y, t):
    return y - t

# Gradient function for the weight parameter at the output layer
def gradient_weight_out(h, grad_output):
    return h * grad_output

def gradient_hidden(wo, grad_output):
    return wo * grad_output

def gradient_weight_hidden(x, zh, h, grad_hidden):
    return x * -2 * zh * h * grad_hidden

# Function for one iteration of backprop
def backprop_update(x, t, wh, wo, learning_rate):
    """
    Compute the output of the network. This can be done with y = nn(x, wh, wo),
    but we need the intermediate h and zh for the weight updates
    """
    zh = x * wh
    h = rbf(zh)     # hidden_activations(x, wh)
    y = output_activations(h, wo)           # network output c
    # Compute the gradient at the output
    grad_output = gradient_output(y, t)
    # get the delta for wo
    d_wo = learning_rate * gradient_weight_out(h, grad_output)
    # Compute the gradient at the hidden layer
    grad_hidden = gradient_hidden(wo, grad_output)
    # get the delta for wh
    d_wh = learning_rate * gradient_weight_hidden(x, zh, h, grad_hidden)
    # compute the update parameters
    return (wh - d_wh.sum(), wo - d_wo.sum())       # TODO : Check the types here, use np.sum()?


# Train the network. This network consists of a single neuron
def train(x, t, verbose=True):
    wh = 2
    wo = -5
    learning_rate = 0.2

    num_iter = 50
    lr_update = learning_rate / num_iter    # Smooth the learning rate over time
    w_cost_iter = [(wh, wo, cost_for_param(x, wh, wo, t))]      # list to store weight values over the iterations

    for i in range(num_iter):
        learning_rate -= lr_update
        wh, wo = backprop_update(x, t, wh, wo, learning_rate)
        w_cost_iter.append((wh, wo, cost_for_param(x, wh, wo, t)))
    # Print the final cost
    if(verbose):
        print("Final cost is %.2f for weights wh : %.2f and wo: %.2f" % (cost_for_param(x, wh, wo, t), wh, wo))

    return w_cost_iter

def compute_cost_weights(x, t, nb_of_ws):
    wsh = np.linspace(-10, 10, num=nb_of_ws)
    wso = np.linspace(-10, 10, num=nb_of_ws)
    ws_x, ws_y = np.meshgrid(wsh, wso)

    cost_ws = np.zeros((nb_of_ws, nb_of_ws))

    # TODO : Move this out so that we can compute various costs and compare
    for i in range(nb_of_ws):
        for j in range(nb_of_ws):
            cost_ws[i,j] = cost(nn(x, ws_x[i,j], ws_y[i,j]), t)

    return (cost_ws, ws_x, ws_y)

# ==== PLOTTING ==== #
def plot_weight_cost(cost_ws, ws_x, ws_y, nb_of_ws):
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(ws_x, ws_y, cost_ws, linewidth=0, cmap=plt.cm.pink)
    ax.view_init(elev=60, azim=-30)
    cbar = fig.colorbar(surf)
    ax.set_xlabel('$w_h$', fontsize=15)
    ax.set_ylabel('$w_o$', fontsize=15)
    ax.set_zlabel('$\\xi$', fontsize=15)
    cbar.ax.set_ylabel('$\\xi$', fontsize=15)
    plt.title('Cost funciton surface')
    plt.grid()
    plt.show()


def plot_weight_cost_and_updates(cost_ws, w_cost_iter, ws_x, ws_y):
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(ws_x, ws_y, cost_ws, linewidth=0, cmap=plt.cm.pink)
    ax.view_init(elev=60, azim=-30)
    cbar = fig.colorbar(surf)
    cbar.ax.set_ylabel('$\\xi$', fontsize=15)

    # Plot the updates
    for i in range(1, len(w_cost_iter)):
        wh1, wo1, c1 = w_cost_iter[i-1]
        wh2, wo2, c2 = w_cost_iter[i]
        # Plot the weight cost update and the line representing the update
        ax.plot([wh1], [wo1], [c1], 'w+')   # the weight cost
        ax.plot([wh1, wh2], [wo1, wo2], [c1, c2], 'w-')

    # Plot the last weights
    wh1, wo1, c1 = w_cost_iter[len(w_cost_iter)-1]
    ax.plot([wh1], [wo1], c1, 'w+')

    ax.set_xlabel('$w_h$', fontsize=15)
    ax.set_ylabel('$w_o$', fontsize=15)
    ax.set_zlabel('$\\xi$', fontsize=15)
    plt.title('Gradient descent updates on cost surface')
    plt.grid()
    plt.show()


# ======== ENTRY POINT ======== #
if __name__ == '__main__':

    # set seed for reproductability
    np.random.seed(seed=1)

    nb_of_ws = 200
    #Define and generate the samples
    nb_of_samples_per_class = 20  # The number of sample in each class
    blue_mean = [0]  # The mean of the blue class
    red_left_mean = [-2]  # The mean of the red class
    red_right_mean = [2]  # The mean of the red class

    std_dev = 0.5  # standard deviation of both classes
    # Generate samples from both classes
    x_blue = np.random.randn(nb_of_samples_per_class, 1) * std_dev + blue_mean
    x_red_left = np.random.randn(nb_of_samples_per_class/2, 1) * std_dev + red_left_mean
    x_red_right = np.random.randn(nb_of_samples_per_class/2, 1) * std_dev + red_right_mean

    # Merge samples in set of input variables x, and corresponding set of
    # output variables t
    x = np.vstack((x_blue, x_red_left, x_red_right))
    t = np.vstack((np.ones((x_blue.shape[0],1)),
                np.zeros((x_red_left.shape[0],1)),
                np.zeros((x_red_right.shape[0], 1))))

    # TODO : create plot handles here to allow for programmatic plotting
    # Train the network
    w_cost_iter = train(x, t)
    cost_ws, ws_x, ws_y = compute_cost_weights(x, t, nb_of_ws)
    plot_weight_cost_and_updates(cost_ws, w_cost_iter, ws_x, ws_y)
    #plot_weight_cost(nb_of_ws)
