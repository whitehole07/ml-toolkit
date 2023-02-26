import numpy as np


def red_wine_dataset():
    # Load the CSV file into a NumPy array
    data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)

    # Generate a random permutation of the row indices
    permuted_indices = np.random.permutation(data.shape[0])

    # Shuffle the rows of the array using the permuted indices
    shuffled_X = data[permuted_indices]

    # Normalize all columns of the matrix using L2 normalization
    data_norm = shuffled_X / np.linalg.norm(data, axis=0, ord=2)

    # Determine the number of rows in the training and validation subsets
    n_train = int(0.8 * len(data_norm))
    n_val = len(data_norm) - n_train

    # Split the matrix into training and validation subsets
    data_train = data_norm[:n_train, :]
    data_val = data_norm[n_train:, :]

    # Split dataset
    X_train = data_train[:, :-2]
    Y_train = data_train[:, -1]

    X_val = data_val[:, :-2]
    Y_val = data_val[:, -1]

    return data, X_train, Y_train, X_val, Y_val
