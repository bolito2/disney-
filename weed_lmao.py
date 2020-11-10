from layers import *
from encoding import *

import matplotlib.pyplot as plt

import csv
import sys
import getopt

import random

# Path to save the parameters
filename = 'parameters.h5'


# Train the RNN with the given parameters
def train(learning_rate, units, epochs):
    # Try to load the parameters if they are saved, create a new RNN with the specified units otherwise
    rnn = RNN(filename=filename, units=units)

    # Extract the strain names from the dataset
    with open('cannabis.csv', newline='', encoding="utf-8") as csvfile:
        cannabis_data = csv.reader(csvfile)
        names_oh = []

        # The first column of the data contains the strain name
        for row in cannabis_data:
            # Replace syphons with spaces
            name = row[0].replace('-', ' ').lower()

            # Add the end token to the name
            name = name + '>'

            # Convert to one-hot vector and append to the array
            names_oh.append(one_hot_string(name))

        # First row is metadata so delete it
        names_oh = names_oh[1:]

    # Keep track of the average cost in each epoch
    costs = []

    for e in range(epochs):
        cost = 0
        for name_oh in names_oh:
            # Apply forward-propagation
            cost += rnn(name_oh)

            # Backpropagate and update weights of the RNN
            rnn.backpropagate()
            rnn.update_weights(learning_rate)

        cost /= len(names_oh)

        print('Epoch {}: J = {}'.format(e + 1, cost))
        costs.append(cost)

    # Save the updated parameters
    rnn.save_parameters(filename)

    # Plot the cost in each epoch
    plt.plot(costs, color='r')
    plt.show()


# Generate a name with the trained RNN
def gen_names(filename):

    print('Input \\ to exit')

    # Load the RNN from file
    rnn = RNN(filename=filename)

    while True:
        letter = input('Input first letter(leave blank for random letter): ')

        if letter == '\\':
            return

        if letter == '':
            letter = letters[random.randint(11, 36)]
        gen_strain = ''

        rnn_cell = rnn.rnn_cell

        a = np.zeros((10, 1))
        while letter != '>':
            # Add last letter to the name of the strain
            gen_strain += letter

            # Forward-propagate one step
            a, y = rnn_cell(one_hot_character(letter), a)

            # Get the probabilities of choosing each character
            probabilities = np.reshape(softmax(y), [-1])

            letter_index = np.random.choice(n_letters, p=probabilities)
            letter = letters[letter_index]

        print(gen_strain)


def train_args(arg_list):
    opts, arga = getopt.getopt(arg_list, 'r:u:e:')
    learning_rate = 0.05
    units = 10
    epochs = 50

    for opt, value in opts:
        if opt == '-r':
            learning_rate = float(value)
        if opt == '-u':
            units = int(value)
        if opt == '-e':
            epochs = int(value)

    train(learning_rate, units, epochs)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train_args(sys.argv[2:])
    if sys.argv[1] == 'generate':
        gen_names(filename)


