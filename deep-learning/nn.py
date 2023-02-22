import numpy as np

from activation_functions import relu


class Neuron(object):
    def __init__(self, input_size: int, index: int, input_neuron=False):
        # Location
        self.index = index

        # Parameters
        if not input_neuron:
            self.w = 0.1 * np.random.rand(1, input_size)  # weights
        else:
            self.w = np.ones(input_size)  # weights
        self.b = 0                        # bias


class Layer(object):
    def __init__(self, input_size: int, layer_size: int, input_layer=False):
        # Init neurons
        self.input_size = input_size
        self.layer_size = layer_size

        self.neurons = [Neuron(input_size, index, input_neuron=input_layer) for index in range(layer_size)]  # list of neurons

        # Generate parameters arrays
        self.W = np.vstack((nj.w for nj in self.neurons))
        self.B = np.vstack((nj.b for nj in self.neurons))

        # Misc
        self.__input_layer = input_layer

    def out(self, z: np.ndarray):
        # Compute layer output given input z
        if not self.__input_layer:
            return relu(np.dot(self.W, z) + self.B)
        else:
            return z


class FFNN(object):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        # Hyper-parameters
        self.alpha = None       # Learning rate
        self.batch_size = None  # Batch size
        self.epochs = None      # Number of epochs

        # Train sets
        self.x_train = x_train  # Size of (samples, features)
        self.y_train = y_train  # Size of (samples, outputs)

        # Init Loss list
        self.losses = []
        self.MSE = None

        # Sizes
        self.n_features = x_train.shape[1]
        self.n_outputs = y_train.shape[1]
        self.hidden_size = []

        # Layers
        self.layers = [Layer(self.n_features, self.n_features, input_layer=True)]  # Init Input Layer

    def add_layer(self, layer_size: int):
        # Find input size
        input_size = self.layers[-1].layer_size

        # Add layer
        self.layers.append(Layer(input_size, layer_size))
        self.hidden_size.append(layer_size)

    def train(self, *, alpha=0.1, batch_size=32, epochs=100):
        # Set hyper-parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs

        # Init losses
        self.losses = []

        # Mini-batches Stochastic Gradient Descent (SGD)
        # Train for the specified number of epochs
        for epoch in range(epochs):
            # Shuffle the training data
            idx = np.random.permutation(self.x_train.shape[0])
            X_shuffle, y_shuffle = self.x_train[idx], self.y_train[idx]

            # Divide the training data into mini-batches
            num_batches = int(np.ceil(self.x_train.shape[0] / batch_size))
            X_batches = np.array_split(X_shuffle, num_batches)
            y_batches = np.array_split(y_shuffle, num_batches)

            # Train on each mini-batch
            for X_batch, y_batch in zip(X_batches, y_batches):
                # Forward propagation
                y_pred_batch = np.array([])
                activations_batch = np.array([])
                for X_row, y_row in zip(X_batch, y_batch):  # Iterate over batch samples
                    activations = self.__feed_forward(X_row)  # Propagate single sample

                    # Update batch prediction
                    activations_batch = np.append(activations_batch, activations)
                    y_pred_batch = np.append(y_pred_batch, activations[-1])

                # Backpropagation
                for activations_batch_row, y_pred_batch_row, y_batch_row in zip(activations_batch, y_pred_batch, y_batch):
                    self.__back_propagation(y_pred_batch_row, y_batch_row, activations_batch_row)

            # Evaluate MSE at end of epoch
            # self.MSE = np.mean((1 / self.n_outputs) * np.sum((y_pred - y_batch) ** 2, axis=1))
            # self.losses.append(self.MSE)

    def __feed_forward(self, z_in: np.ndarray):
        if len(self.layers) < 2:
            raise ValueError("Output layer was not set, add hidden layer")

        # Feed Forward
        z = z_in  # Init state
        activations = []
        for layer in self.layers:
            z = layer.out(z)       # Propagate state
            activations.append(z)  # Update activations
        return activations  # Network Activations

    def __back_propagation(self, y_pred, y_batch, activations):



        delta = y_pred - y_batch
        for i in range(len(weights) - 1, -1, -1):
            dz = np.where(activations[i + 1] > 0, delta, 0)
            dw = np.dot(activations[i].T, dz) / self.batch_size
            db = np.mean(dz, axis=0, keepdims=True)
            delta = np.dot(dz, weights[i].T)
            weights[i] -= self.alpha * dw
            biases[i] -= self.alpha * db
