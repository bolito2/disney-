import csv
import numpy as np
from layers import *


# Letters supported by the algorithm
letters = ' 0123456789abcdefghijklmnopqrstuvwxyz>'
n_letters = len(letters)


# Returns the one-hot vector encoding of any given letter, using the above dictionary
def one_hot(character):
    return np.array([character == letters[i] for i in range(n_letters)], dtype=np.int32)


# Extract the strain names from the dataset
with open('cannabis.csv', newline='') as csvfile:
    cannabis_data = csv.reader(csvfile)
    cannabis_names = []

    # The first column of the data contains the strain name
    for row in cannabis_data:
        cannabis_names.append(row[0])

    # First row is metadata so delete it
    cannabis_names = cannabis_names[1:]

# Encode names into a numpy array(as one hot matrices)
X = []

for name in cannabis_names:
    # Replace syphons with spaces
    name = name.replace('-', ' ').lower()

    # Add the end token to the name
    name = name + '>'

    # Get one-hot encoding of every character
    name_oh = np.zeros((len(name), n_letters))
    for i in range(len(name)):
        name_oh[i] = one_hot(name[i])

    X.append(name_oh)

letter = 'o'
gen_strain = ''

rnn_cell = RNNCell()
softmax = SoftMax()

a = np.zeros((n_letters, 1))
while letter != '>':
    # Add last letter to the name of the strain
    gen_strain += letter

    # Forward-propagate one step
    a, y = rnn_cell(a, one_hot(letter))

    # Get the probabilities of choosing each character
    probabilities = np.reshape(softmax(y), [-1])

    letter_index = np.random.choice(n_letters, p=probabilities)
    letter = letters[letter_index]

print(gen_strain)
