import numpy as np

# Letters supported by the algorithm
letters = ' 0123456789abcdefghijklmnopqrstuvwxyz>'
n_letters = len(letters)


# Returns the one-hot row vector encoding of any given letter, using the above dictionary
def one_hot_character(character):
    one_hot_list = np.array([character == letters[i] for i in range(n_letters)], dtype=np.int32)
    return np.reshape(one_hot_list, [-1, 1])


# Return the one hot encoding of a string as a matrix (len x n_letters)
def one_hot_string(string):
    string_oh = np.zeros((len(string), n_letters))
    for i in range(len(string)):
        string_oh[i] = one_hot_character(string[i]).squeeze()

    return string_oh
