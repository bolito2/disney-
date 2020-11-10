import csv
import numpy as np
from layers import *
from encoding import *
import matplotlib.pyplot as plt

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


# Create the RNN
rnn = RNN(4)
epochs = 50

# Keep track of the average cost in each epoch
costs = []

for e in range(epochs):
    cost = 0
    for name_oh in names_oh:
        # Apply forward-propagation
        cost += rnn(name_oh)

        # Backpropagate and update weights of the RNN
        rnn.backpropagate()
        rnn.update_weights(0.00002)

    cost /= len(names_oh)

    print('Epoch {}: J = {}'.format(e, cost))
    costs.append(cost)

# Plot the cost in each epoch
plt.plot(costs)
plt.show()


# Generate a name with the trained RNN
def gen_name():
    letter = input('Input first letter:')
    gen_strain = ''

    rnn_cell = rnn.rnn_cell

    a = np.zeros((n_letters, 1))
    while letter != '>':
        # Add last letter to the name of the strain
        gen_strain += letter

        # Forward-propagate one step
        a = rnn_cell(a, one_hot_character(letter))

        # Get the probabilities of choosing each character
        probabilities = np.reshape(softmax(a), [-1])

        letter_index = np.random.choice(n_letters, p=probabilities)
        letter = letters[letter_index]

    print(gen_strain)
