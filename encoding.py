import numpy as np

# Letters supported by the algorithm
letters = ' abcdefghijklmnopqrstuvwxyz>'
n_letters = len(letters)


# Returns the one-hot row vector encoding of any given letter, using the above dictionary
def one_hot_character(character):
    one_hot_list = np.array([character == letters[i] for i in range(n_letters)], dtype=np.int32)
    return np.reshape(one_hot_list, [-1, 1])


# Return the one hot encoding of a string as a matrix (len x n_letters)
# If it has any invalid characters(numbers return that it is invalid)
def one_hot_string(string):
    string_oh = np.zeros((len(string), n_letters))
    for i in range(len(string)):
        if string[i] not in letters:
            return False, None
        string_oh[i] = one_hot_character(string[i]).squeeze()

    return True, string_oh


# Decode a string from one-hot representation
def decode_one_hot(oh):
    word = ''
    for i in range(oh.shape[0]):
        indices = np.where(oh[i] == 1)
        word += letters[indices[0][0]]

    return word
