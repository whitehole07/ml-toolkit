import numpy as np
from matplotlib import pyplot as plt

from deep_learning.nn import FFNN

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

# Init neural network
nn = FFNN(X_train, Y_train, X_val, Y_val, "relu")
nn.add_layer(16)
nn.add_layer(32)

# Train neural network
nn.print_network()
nn.train(alpha=0.001, epochs=45)
nn.plot_mse()

# Validate model
Y_pred_val = nn.predict(X_val)
print("MSE: ", np.mean((np.sum((Y_pred_val - Y_val.reshape(-1, 1))**2, axis=1))))

# Plot the predicted output against the true output
Y_val = Y_val.reshape(-1, 1) * np.linalg.norm(data, axis=0, ord=2)[-1]
Y_pred_val = Y_pred_val * np.linalg.norm(data, axis=0, ord=2)[-1]

plt.plot(Y_val, Y_pred_val, 'bo')
plt.plot([0, 10], [0, 10], 'r--')
plt.title('Validation set prediction')
plt.xlabel('True output')
plt.ylabel('Predicted output')
plt.axis([0, 10, 0, 10])
plt.show()
