"""
LINEAR CLASSIFIER

"""

import numpy as np
from typing import Tuple
from typing import Union


class LinearClassifier:
    def __init__(self, reg:float=5e-6, ss:float=1e-3) -> None:
        self.step_size :float = ss
        self.reg       :float = reg       # regularization strength
        self.num_iter  :int   = 200
        self.W         :np.ndarray = None
        self.scores    :np.ndarray = None
        self.dscores   :np.ndarray = None
        self.loss      :np.ndarray = None

    def init_params(self, D:int, K:int) -> None:
        # K = number of classes
        # D = dimension of data

        self.W = 0.01 * np.random.randn(D, K)
        self.b = np.zeros((1,K))

    def forward_iter(self, X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_examples = X.shape[0]
        self.scores = np.dot(X, self.W) + self.b

        # Compute class probs
        exp_scores = np.exp(self.scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss        = np.sum(correct_logprobs) / num_examples
        reg_loss         = 0.5 * self.reg * np.sum(self.W * self.W)
        loss             = data_loss + reg_loss

        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # Store internal data
        self.loss = loss
        self.dscores = dscores

        return (loss, dscores)

    def backward_iter(self, X:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # backprop the gradient params
        dW = np.dot(X.T, self.dscores)
        db = np.sum(self.dscores, axis=0, keepdims=True)

        dW += self.reg * self.W

        self.W -= self.step_size * dW
        self.b -= self.step_size * db

        return (self.W, self.b)
