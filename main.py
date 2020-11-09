import csv
import numpy as np
from layers import *
from encoding import *
import matplotlib.pyplot as plt

# Extract the strain names from the dataset
with open('cannabis.csv', newline='') as csvfile:
    cannabis_data = csv.reader(csvfile)
    cannabis_names = []

    # The first column of the data contains the strain name
    for row in cannabis_data:
        cannabis_names.append(row[0])

    # First row is metadata so delete it
    cannabis_names = cannabis_names[1:]

# Create the RNNChain
rnn = RNNChain()
epochs = 5

costs = []

for e in range(epochs):
    cost = 0
    for name in cannabis_names:
        # Replace syphons with spaces
        name = name.replace('-', ' ').lower()

        # Add the end token to the name
        name = name + '>'

        # Convert to one-hot vector
        name_oh = one_hot_string(name)

        # Apply forward-propagation
        cost += rnn(name_oh)

        # Backpropagate and update weights of the RNN
        rnn.backpropagate()
        rnn.update_weights(0.005)

    print('Epoch {}: J = {}'.format(e, cost))
    costs.append(cost/len(cannabis_names))

plt.plot(costs)

letter = 'o'
gen_strain = ''

rnn_cell = RNNCell()

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
