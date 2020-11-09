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
    def __call__(self, x, a_prev):
        z = np.dot(self.Wa, a_prev) + np.dot(self.Wx, x) + self.b
        # Use hyperbolic tangent for activation
        a = np.tanh(z)

        return a

    # Compute the weight gradients from the dJ/da
    # x and a are column vectors, da is a row vector(gradient)
    def gradients(self, x, a, da):
        # dJ/dz, computed with element-wise multiplication. It would actually be a diagonal matrix with
        # m_ii = (1 + ai)(1 - ai) times da(row vector), which is equivalent but less efficient.
        dz = np.multiply(np.multiply(1 + a, 1 - a).T, da)   # (1 x n)

        # dz/db = I so they are equal, but we transpose it to make a column vector(for gradient descent)
        db = dz.T   # (n x 1)

        # dJ/dWi = dzi*x.T where dWi is the i-th row of Wx so we can compute them all at the same time this way
        dWx = np.dot(dz.T, x.T)   # (n x n)
        # Same thing for the hidden state weights
        dWa = np.dot(dz.T, a.T)   # (n x n)

        # Compute the gradient with respect to last time-step hidden values, to use for backpropagation through time
        da_prev = np.dot(dz, self.Wa)   # (1 x n)

        # Return the gradients
        return da_prev, dWx, dWa, db


def softmax(a):
    # Exp and normalize the output to convey probabilities
    p = np.exp(a)
    p /= np.sum(p)

    return p


class RNNChain:
    # Class for caching the various outputs of forward-propagation
    class Cache:
        def __init__(self, shape):
            self.X = np.zeros(shape)
            self.A = np.zeros(shape)
            self.P = np.zeros(shape)

    # Class for storing the various gradients of back-propagation
    class Gradients:
        def __init__(self):
            self.db = np.zeros((n_letters, 1))
            self.dWx = np.zeros((n_letters, n_letters))
            self.dWa = np.zeros((n_letters, n_letters))

    def __init__(self):
        self.rnn_cell = RNNCell()
        self.cache = None
        self.long = None
        self.grads = None

    # Forward-propagate a given name and return predictions
    def __call__(self, X):
        # Prepare the cache
        self.cache = self.Cache(X.shape)
        # Cache the inputs for backpropagation
        self.cache.X = X

        # Set the longitude of the word minus the end token (longitude of the RNN)
        self.long = X.shape[0] - 1

        # Initialize the cost to zero
        cost = 0

        # Start the hidden values as zeros
        a = np.zeros((n_letters, 1))

        for t in range(self.long):
            # Get the inputs in proper shape(n x 1)
            x = np.reshape(X[t], [-1, 1])

            # Process one step of forward propagation to get the hidden values of the i-th layer(indexed from 1) from the last
            a = self.rnn_cell(x, a)

            # Transpose a before saving it into the cache, as it is a column vector. cache.A is indexed from one
            self.cache.A[t + 1] = a.T

            # Get the predictions by applying soft-max to this layer's hidden weights and cache them too
            self.cache.P[t + 1] = softmax(a).T

            # Compute cost of this time-step and add it to the total
            # The expected value is the next character from the input
            cost += -np.sum(np.multiply(X[t + 1], np.log(self.cache.P[t + 1])))

        return cost/self.long

    # Apply backpropagation through time to get the gradients
    def backpropagate(self):
        if self.long is None:
            print('Error: Apply forward propagation before back propagation')
        else:
            # Initialize the gradients
            self.grads = self.Gradients()

            # Iterate through layers 1 to long(included)
            for t in range(1, self.long + 1):
                # Compute dJ/da where J is the cost of the time-step t and a its hidden state(da has to be a row vector)
                da = self.cache.P[t, None] - self.cache.X[t, None]

                # Here is where we backpropagate through time, getting the gradient of dJ with respect to the previous
                # layer hidden values and updating the weights gradients each step
                for j in range(t, 0, -1):
                    # Compute the gradients of each time-step keeping in mind that a and x must be column vectors
                    da, dWx_j, dWa_j, db_j = self.rnn_cell.gradients(self.cache.X[j, None].T, self.cache.A[j, None].T, da)

                    self.grads.dWx += dWx_j
                    self.grads.dWa += dWa_j
                    self.grads.db += db_j

