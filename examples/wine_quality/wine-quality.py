import numpy as np

from deep_learning.nn import FFNN

# Load the CSV file into a NumPy array
data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)

# Normalize all columns of the matrix using L2 normalization
data_norm = data / np.linalg.norm(data, axis=0, ord=2)

# Split dataset
X_train = data_norm[:, :-2]
Y_train = data_norm[:, -1]

# Init neural network
nn = FFNN(X_train, Y_train)
nn.add_layer(16)

# Train neural network
nn.train()





