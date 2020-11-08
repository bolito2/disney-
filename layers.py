import numpy as np

# Letters supported by the algorithm
letters = ' 0123456789abcdefghijklmnopqrstuvwxyz>'
n_letters = len(letters)


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
        # Reshape x to a row vector in case it only has one dimension
        x = np.reshape(x, [-1, 1])

        a = np.dot(self.Wa, a_prev) + np.dot(self.Wx, x) + self.b
        # Use hyperbolic tangent for activation
        y = np.tanh(a)

        return a, y


class SoftMax:
    def __call__(self, y):
        # Exp and normalize the output to convey probabilities
        y_softmax = np.exp(y)
        y_softmax /= np.sum(y_softmax)

        return y_softmax
