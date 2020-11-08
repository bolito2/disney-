import numpy as np

# Letters supported by the algorithm
letters = ' 0123456789abcdefghijklmnopqrstuvwxyz'
n_letters = len(letters)


class RNNCell:
    def __init__(self, units):
        # Bias vector
        self.b = np.zeros((units, 1))

        # Weights, take into account the last layer weights(units) and input weights(n_letters)
        # They have to be randomly-initialized, unlike the bias vector
        self.Wx = np.random.random((units, n_letters))
        self.Wa = np.random.random((units, units))

    # Forward propagate one step
    def __call__(self, a_prev, x):
        a = np.dot(self.Wa, a_prev) + np.dot(self.Wx, x) + self.b
        # Use hyperbolic tangent for activation
        y = np.tanh(a)

        return a, y
