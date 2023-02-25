import struct
import numpy as np

import matplotlib.pyplot as plt

from deep_learning.pre_processing import one_hot_encode
from deep_learning.nn import FFNN


def read_idx(filename):
    """
    This function reads the IDX file format used by the MNIST dataset.
    """
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>i', f.read(4))[0]
        num_items = struct.unpack('>i', f.read(4))[0]
        if magic_number == 2051:
            num_rows = struct.unpack('>i', f.read(4))[0]
            num_cols = struct.unpack('>i', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((num_items, num_rows, num_cols))
            return data
        elif magic_number == 2049:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data


# Load the MNIST dataset
x_train = read_idx('train-images.idx3-ubyte')
y_train = read_idx('train-labels.idx1-ubyte')
x_test = read_idx('t10k-images.idx3-ubyte')
y_test = read_idx('t10k-labels.idx1-ubyte')

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten images to be 1D arrays
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# Encode
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Init neural network
nn = FFNN(x_train, y_train, x_test, y_test, "relu")
nn.add_layer(16)

# Train neural network
nn.print_network()
nn.train(alpha=0.001, batch_size=6000, epochs=45)
nn.plot_mse()

# Validate model
Y_pred_val = nn.predict(x_test)
print("MSE: ", np.mean((np.sum((Y_pred_val - y_test)**2, axis=1))))

# Plot the predicted output against the true output
plt.plot(y_test, Y_pred_val, 'bo')
plt.plot([0, 10], [0, 10], 'r--')
plt.title('Validation set prediction')
plt.xlabel('True output')
plt.ylabel('Predicted output')
plt.axis([0, 10, 0, 10])
plt.show()
