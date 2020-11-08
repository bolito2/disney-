import numpy as np
from encoding import *


class RNNCell:
    def __init__(self):
        # Bias vector
        self.b = np.zeros((n_letters, 1))

        # Set the weights, for both the last layer activations and this layers inputs
        # They have to be randomly-initialized, unlike the bias vector
        self.Wx = np.random.random((n_letters, n_letters))
        self.Wa = np.random.random((n_letters, n_letters))

    # Forward propagate one step
    def __call__(self, a_prev, x):
        z = np.dot(self.Wa, a_prev) + np.dot(self.Wx, x) + self.b
        # Use hyperbolic tangent for activation
        a = np.tanh(z)

        return a


def softmax(a):
    # Exp and normalize the output to convey probabilities
    y = np.exp(a)
    y /= np.sum(y)

    return y


class RNNChain:
    def __init__(self):
        self.rnn_cell = RNNCell()
        self.A = None
