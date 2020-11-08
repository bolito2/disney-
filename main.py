import csv
import numpy as np

# Letters supported by the algorithm
letters = ' 0123456789abcdefghijklmnopqrstuvwxyz'
n_letters = len(letters)

# Extract the strain names from the dataset
with open('cannabis.csv', newline='') as csvfile:
    cannabis_data = csv.reader(csvfile)
    cannabis_names = []

    # The first column of the data contains the strain name
    for row in cannabis_data:
        cannabis_names.append(row[0])

    # First row is metadata so delete it
    cannabis_names = cannabis_names[1:]

# Encode names in a one-hot way
for name in cannabis_names:
    name_oh = np.zeros((len(name), n_letters))

    # Replace syphons with spaces
    name = name.replace('-', ' ').lower()

    for i in range(len(name)):
        character = name[i]
        # One-hot encoding of very single character
        letter_oh = [character == letters[i] for i in range(n_letters)]
        name_oh[i] = letter_oh

