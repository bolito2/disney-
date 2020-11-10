import numpy as np
from encoding import *
import h5py

class RNNCell:
    def __init__(self, units):
        # Bias vectors
        self.ba = np.zeros((units, 1))
        self.by = np.zeros((n_letters, 1))

        # Set the weights, for the last layer activations(a) and this layers inputs(x), and also to get the outputs(y)
        # They have to be randomly-initialized, unlike the bias vector
        self.Wx = np.random.random((units, n_letters))*np.sqrt(2/n_letters)
        self.Wa = np.random.random((units, units))*np.sqrt(2/n_letters)
        self.Wy = np.random.random((n_letters, units))*np.sqrt(2/n_letters)

    # Forward propagate one step
    def __call__(self, x, a_prev):
        z = np.dot(self.Wa, a_prev) + np.dot(self.Wx, x) + self.ba
        # Use hyperbolic tangent for activation
        a = np.tanh(z)
        # Compute outputs
        y = np.dot(self.Wy, a) + self.by

        return a, y

    # Compute the gradients of the 'hidden' weights(Wx, Wa, ba) and return the gradient of last time-step for backprop

    # Inputs:
    # x -> Input of this time-step (n x 1)
    # a_prev -> Hidden state of last time-step (units x 1)
    # a -> Hidden state of this time-step (units x 1)
    # da -> Gradient of dJ/da, where a is the hidden values of this time-step (1 x units)

    # Outputs:
    # da_prev -> Cost gradient with respect to the previous time-step hidden values (1 x units)
    # dWx -> Gradient of the cost with respect to the Wx matrix, in the same dimension (units x n)
    # dWa -> Gradient of the cost with respect to the Wa matrix, in the same dimension (units x units)
    # dba -> Gradient of the cost with respect to the ba bias vector, in the same dimension (units x 1)
    def hidden_gradients(self, x, a_prev, a, da):
        # dJ/dz, computed with element-wise multiplication. It would actually be a diagonal matrix with
        # m_ii = (1 + ai)(1 - ai) times da(row vector), which is equivalent but less efficient.
        dz = np.multiply(1 - np.square(a).T, da)   # (1 x units)

        # dz/db = I so they are equal, but we transpose it to make a column vector(for gradient descent)
        dba = dz.T   # (units x 1)

        # dJ/dWi = dzi*x.T where dWi is the i-th row of Wx so we can compute them all at the same time this way
        dWx = np.dot(dz.T, x.T)   # (units x n)
        # Same thing for the hidden state weights from last time-step
        dWa = np.dot(dz.T, a_prev.T)   # (units x units)

        # Compute the gradient with respect to last time-step hidden values, to use for backpropagation through time
        da_prev = np.dot(dz, self.Wa)   # (1 x units)

        # Return the gradients
        return da_prev, dWx, dWa, dba

    # Compute the gradients of the 'output' weights(Wy, by) and return the gradient with respect to the hidden weights

    # Inputs:
    # a -> Hidden state of this time-step (units x 1)
    # dy -> Gradient of dJ/dy, where y is the output of this time-step (1 x n)

    # Outputs:
    # da -> Cost gradient with respect to the hidden values of this time-step (1 x units)
    # dWy -> Gradient of the cost with respect to the Wy matrix, in the same dimension (n x units)
    # dby -> Gradient of the cost with respect to the by bias vector, in the same dimension (n x 1)
    def output_gradients(self, a, dy):
        # dy/db = I so they are equal, but we transpose it to make a column vector(for gradient descent)
        dby = dy.T  # (n x 1)
        # dJ/dWi = dyi*a.T where dWi is the i-th row of Wy so we can compute them all at the same time this way
        dWy = np.dot(dy.T, a.T)  # (n x units)

        # dJ/da, by multiplying dJ/dy by dy/da = Wy
        da = np.dot(dy, self.Wy)  # (1 x units)

        # Return the gradients
        return da, dWy, dby


def softmax(a):
    # Exp and normalize the output to convey probabilities
    p = np.exp(a)
    p /= np.sum(p)

    return p


class RNN:
    # Class for caching the various outputs of forward-propagation
    class Cache:
        def __init__(self, units, long):
            # We use long + 1 because there is one more character than layers in the RNN(the end token >)
            self.X = np.zeros((long + 1, n_letters))
            self.A = np.zeros((long + 1, units))
            self.P = np.zeros((long + 1, n_letters))

    # Class for storing the various gradients of back-propagation
    class Gradients:
        def __init__(self, units):
            # Biases
            self.dba = np.zeros((units, 1))
            self.dby = np.zeros((n_letters, 1))

            # Weight matrices
            self.dWx = np.zeros((units, n_letters))
            self.dWa = np.zeros((units, units))
            self.dWy = np.zeros((n_letters, units))

    def __init__(self, units=None, filename=None):
        if filename is None:
            if units is None:
                raise ValueError('You have to specify the number of units of the RNN if it isn\'t loaded from a file.')

            self.units = units
            self.rnn_cell = RNNCell(units)
            self.cache = None
            self.long = None
            self.grads = None
        else:
            try:
                with h5py.File(filename, 'r') as f:
                    # Had to save units as a 1-dimensional array of size 1 lmao
                    self.units = f['units'][0]

                    self.rnn_cell = RNNCell(self.units)
                    self.cache = None
                    self.long = None
                    self.grads = None

                    self.rnn_cell.Wa = np.array(f['Wa'])
                    self.rnn_cell.Wx = np.array(f['Wx'])
                    self.rnn_cell.Wy = np.array(f['Wy'])

                    self.rnn_cell.ba = np.array(f['ba'])
                    self.rnn_cell.by = np.array(f['by'])
            except OSError:
                raise FileNotFoundError('Can\'t find the parameters. Download them or train the network.')

    # Forward-propagate a given name and cache the internal state of the RNN
    def __call__(self, X):
        # Set the longitude of the word minus the end token (longitude of the RNN)
        self.long = X.shape[0] - 1

        # Prepare the cache
        self.cache = self.Cache(self.units, self.long)
        # Cache the inputs for backpropagation
        self.cache.X = X

        # Initialize the cost to zero
        cost = 0

        # Start the hidden values as zeros
        a = np.zeros((self.units, 1))

        for t in range(1, self.long + 1):
            # Get the inputs in proper shape(n x 1)
            x = np.reshape(X[t - 1], [-1, 1])

            # Process one step of forward propagation to get the hidden values of the t time-step(indexed from 1)
            # and the output values of the RNN for predictions
            a, y = self.rnn_cell(x, a)

            # Transpose a before saving it into the cache, as it is a column vector. cache.A is indexed from one
            self.cache.A[t] = a.T

            # Get the predictions by applying soft-max to this layer's output and cache them too
            self.cache.P[t] = softmax(y).T

            # Compute cost of this time-step and add it to the total
            # The expected value is the next character from the input
            cost += -np.sum(np.multiply(X[t], np.log(self.cache.P[t])))

        return cost/self.long

    # Apply backpropagation through time to get the gradients
    def backpropagate(self):
        if self.long is None:
            print('Error: Apply forward propagation before back propagation')
            return

        # Initialize the gradients
        self.grads = self.Gradients(self.units)

        # Iterate through layers 1 to long(included)
        for t in range(1, self.long + 1):
            # Compute dJ/dy where J is the cost of the time-step t and y its output -> (n x 1)
            dy = np.reshape(self.cache.P[t] - self.cache.X[t], [1, -1])/self.long

            # Compute the output weights' gradients and the gradient of the hidden state to get the rest
            da, dWy, dby = self.rnn_cell.output_gradients(np.reshape(self.cache.A[t], [-1, 1]), dy)

            # Update output gradients
            self.grads.dWy += dWy
            self.grads.dby += dby

            # Here is where we backpropagate through time, getting the gradient of dJ with respect to the previous
            # layer hidden values and updating the weights gradients each step
            for j in range(t, max(0, t-3), -1):
                # Compute the hidden weights' gradients of each time-step keeping in mind that a and x must be columns
                da, dWx_j, dWa_j, dba_j = self.rnn_cell.hidden_gradients(np.reshape(self.cache.X[j - 1], [-1, 1]), np.reshape(self.cache.A[j - 1], [-1, 1]), np.reshape(self.cache.A[j], [-1, 1]), da)

                # Update hidden gradients
                self.grads.dWx += dWx_j
                self.grads.dWa += dWa_j
                self.grads.dba += dba_j

    # Update the weights with the gradients
    def update_weights(self, learning_rate, clipnorms=None):
        self.rnn_cell.Wx -= learning_rate*self.grads.dWx
        self.rnn_cell.Wa -= learning_rate*self.grads.dWa
        self.rnn_cell.Wy -= learning_rate*self.grads.dWy

        self.rnn_cell.ba -= learning_rate * self.grads.dba
        self.rnn_cell.by -= learning_rate * self.grads.dby

    # Save the parameters in the file that you choose
    def save_parameters(self, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('Wa', data=self.rnn_cell.Wa)
            f.create_dataset('Wx', data=self.rnn_cell.Wx)
            f.create_dataset('Wy', data=self.rnn_cell.Wy)

            f.create_dataset('ba', data=self.rnn_cell.ba)
            f.create_dataset('by', data=self.rnn_cell.by)

            f.create_dataset('units', data=np.array([self.units]))

    # And load them
    def load_parameters(self, filename):
        with h5py.File(filename, 'r') as f:
            self.rnn_cell.Wa = np.array(f['Wa'])
            self.rnn_cell.Wx = np.array(f['Wx'])
            self.rnn_cell.Wy = np.array(f['Wy'])

            self.rnn_cell.ba = np.array(f['ba'])
            self.rnn_cell.by = np.array(f['by'])
