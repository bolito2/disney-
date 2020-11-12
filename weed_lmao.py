from layers import *
from encoding import *

import matplotlib.pyplot as plt

import csv
import sys
import getopt

import random

# Path to save the parameters
filename = 'parameters.npz'


# Train the RNN with the given parameters
def train(learning_rate, units, epochs):
    # Try to load the parameters if they are saved, create a new RNN with the specified units otherwise
    rnn = RNN(filename=filename, units=units)

    # Extract the strain names from the dataset
    with open('cannabis.csv', newline='', encoding="utf-8") as csvfile:
        cannabis_data = csv.reader(csvfile)
        names_oh = []
        excluded_names = 0

        print('Loading weed strain names from database...')
        # The first column of the data contains the strain name
        for row in cannabis_data:
            # Replace syphons with spaces
            name = row[0].replace('-', ' ').lower()

            # Add the end token to the name
            name = name + '>'

            # Convert to one-hot vector and append to the array
            valid, name_oh = one_hot_string(name)
            # Only append the name if it's valid(no numbers in it)
            if valid:
                names_oh.append(name_oh)
            else:
                excluded_names += 1

        # First row is metadata so delete it
        names_oh = names_oh[1:]

        print('{} names were excluded because they contained numbers or other invalid characters. {} names remain.'.format(excluded_names, len(names_oh)))

    # Keep track of the average cost in each epoch
    costs = []

    print('==============================================')
    print('Training for {} epochs with learning_rate={}'.format(epochs, learning_rate))
    for e in range(epochs):
        cost = 0
        for name_oh in names_oh:
            # Apply forward-propagation
            cost += rnn(name_oh)

            # Backpropagate and update weights of the RNN
            rnn.backpropagate()
            rnn.update_weights(learning_rate)

        cost /= len(names_oh)

        print('(Epoch {}/{}) Cost = {}'.format(e + 1, epochs, cost), end='\r')
        costs.append(cost)

    print('Training finished, Cost: {} -> {}'.format(costs[0], costs[-1]))
    print('==============================================')

    # Save the updated parameters
    rnn.save_parameters(filename)

    # Plot the cost in each epoch
    plt.plot(costs, color='r')

    # Change the name of the window
    fig = plt.gcf()
    fig.canvas.set_window_title('WEED LMAO')

    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.show()


# Generate a name with the trained RNN
def gen_names():
    # Load the RNN from file
    rnn = RNN(filename=filename)

    print('Input how the name should start. Leave blank if you want it completely random and type \\ to exit')

    while True:
        # Get the user's chosen start for the strain name, and lowercase it
        start = input().lower()

        if start == '\\':
            return

        # Start with random letter if no input is given
        if start == '':
            # Only pick a letter, don't start with space or end-token
            start = letters[random.randint(1, n_letters - 2)]

        # Generate the string if the input is valid
        valid, gen_strain = rnn.gen_name(start)

        if valid:
            print(gen_strain)
        else:
            print('Input contains invalid characters. Only use letters a-z and spaces.')


def train_args(arg_list):
    opts, arga = getopt.getopt(arg_list, 'r:u:e:')
    learning_rate = 0.07
    units = 32
    epochs = 100

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
        gen_names()


