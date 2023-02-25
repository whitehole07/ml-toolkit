import numpy as np

from deep_learning.activation_functions import relu, drelu


class Neuron(object):
    def __init__(self, input_size: int, index: int, input_neuron=False):
        # Location
        self.index = index

        # Parameters
        if not input_neuron:
            self.w = 0.1 * np.random.rand(input_size)  # weights
        else:
            self.w = np.ones(input_size)  # weights
        self.b = 0                        # bias


class Layer(object):
    def __init__(self, input_size: int, layer_size: int, input_layer=False):
        # Init neurons
        self.input_size = input_size
        self.layer_size = layer_size

        self.neurons = [Neuron(input_size, index, input_neuron=input_layer) for index in range(layer_size)]  # list of neurons

        # Misc
        self.__input_layer = input_layer

    @property
    def W(self):
        return np.vstack([nj.w for nj in self.neurons])

    @property
    def B(self):
        return np.vstack([nj.b for nj in self.neurons])

    def out(self, z: np.ndarray):
        # Compute layer output given input z
        if not self.__input_layer:
            z = (np.dot(self.W, z.reshape(-1, 1)) + self.B).reshape((-1, ))
            return z, relu(z)
        else:
            return z.reshape((-1, )), z.reshape((-1, ))


class FFNN(object):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        # Hyper-parameters
        self.alpha = None       # Learning rate
        self.batch_size = None  # Batch size
        self.epochs = None      # Number of epochs

        # Train sets
        self.x_train = x_train if x_train.ndim > 1 else x_train.reshape((-1, 1))  # Size of (samples, features)
        self.y_train = y_train if y_train.ndim > 1 else y_train.reshape((-1, 1))  # Size of (samples, outputs)

        # Init Loss list
        self.losses = []
        self.MSE = None

        # Sizes
        self.n_features = x_train.shape[1]
        self.n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        self.hidden_size = []

        # Layers
        self.layers = [Layer(self.n_features, self.n_features, input_layer=True), Layer(self.n_features, self.n_outputs)]  # Init Input Output Layer

    def add_layer(self, layer_size: int):
        # Remove output layer
        del self.layers[-1]

        # Find input size
        input_size = self.layers[-1].layer_size

        # Add layer
        self.layers.append(Layer(input_size, layer_size))
        self.hidden_size.append(layer_size)

        # Add new output layer
        self.layers.append(Layer(layer_size, self.n_outputs))

    def train(self, *, alpha=0.1, batch_size=32, epochs=100):
        # Set hyper-parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs

        # Init losses
        self.losses = []

        # Mini-batches Stochastic Gradient Descent (mSGD)
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
                zs_batch, activations_batch, y_pred_batch = self.__feed_forward(X_batch)  # Propagate single sample

                # Back propagation
                self.__back_propagation(y_pred_batch, y_batch, zs_batch, activations_batch)

                # Evaluate MSE at end of epoch
                y_pred = self.predict(X_batch)
                self.MSE = np.mean((1 / self.n_outputs) * np.sum((y_pred - y_batch) ** 2, axis=1))
                self.losses.append(self.MSE)

    def predict(self, z_in: np.ndarray):
        # Feed Forward
        _, _, pred = self.__feed_forward(z_in)
        return pred  # Network Activations

    def __feed_forward(self, z_in: np.ndarray):
        if len(self.layers) < 2:
            raise ValueError("Output layer was not set, add hidden layer")

        # Feed Forward
        activations_batch = []
        pred = np.array([])
        zs_batch = []
        for a in z_in:
            activations = []
            zs = []
            for layer in self.layers:
                z, a = layer.out(a)    # Propagate state
                zs.append(z)           # Update states
                activations.append(a)  # Update activations

            # Append
            activations_batch.append(activations)
            pred = np.vstack([pred, activations[-1]]) if pred.any() else activations[-1]
            zs_batch.append(zs)
        return zs_batch, activations_batch, pred  # Network Activations

    def __back_propagation(self, y_pred_batch, y_batch, zs_batch, activations_batch):
        # Init gradient matrices
        dCw = []  # first layer skipped, reason for indexing with l-1 instead of l
        dCb = []  # first layer skipped, reason for indexing with l-1 instead of l

        # Iterate over batch samples
        for zs_batch_row, activations_batch_row, y_pred_batch_row, y_batch_row in zip(zs_batch, activations_batch, y_pred_batch, y_batch):
            # Init batch matrices
            dCwl = [[[] for _ in range(len(self.layers[x].neurons))] for x in range(1, len(self.layers))]
            dCbl = [[] for _ in range(1, len(self.layers))]

            # Last layer
            delta_l_next = []
            for j in range(self.layers[-1].layer_size):
                delta_l_next.append((2 / self.n_outputs) * (y_batch_row[j] - y_pred_batch_row[j]) * drelu(zs_batch_row[-1][j]))

                # Compute weight gradients
                for z in range(self.layers[-2].layer_size):
                    dCwl[-1][j].append(delta_l_next[-1] * activations_batch_row[-2][z])  # here j is in layer l and z in layer l-1

            # Add gradients
            dCbl[-1] = delta_l_next

            # Iterate over next layers
            for l in range(len(self.layers)-2, 0, -1):  # Exclude output and input layers
                # Next layer
                delta_l = []
                for j in range(self.layers[l].layer_size):
                    dl = 0
                    for k in range(self.layers[l+1].layer_size):
                        dl += delta_l_next[k] * self.layers[l+1].W[k, j] * drelu(zs_batch_row[l][j])
                    delta_l.append(dl)

                    # Compute weight gradients
                    for z in range(self.layers[l-1].layer_size):
                        dCwl[l-1][j].append(delta_l[-1] * activations_batch_row[l-1][z])  # here j is in layer l and z in layer l-1

                # Add gradients
                dCbl[l-1] = delta_l

                # Clear delta_l_next
                delta_l_next = delta_l

            # Batch completed, update matrix
            dCw.append(dCwl)
            dCb.append(dCbl)

        # Average out across batches
        dCw_avg = self.__avg_gradient_weight(dCw)
        dCb_avg = self.__avg_gradient_bias(dCb)

        # Update parameters
        for l in range(len(self.layers)-1, 0, -1):
            for k in range(self.layers[l].layer_size):
                # Bias
                self.layers[l].neurons[k].b -= self.alpha * dCb_avg[l-1][k]  # Update neuron

                # Weight
                for j in range(self.layers[l-1].layer_size):
                    self.layers[l].neurons[k].w[j] -= self.alpha * dCw_avg[l-1][k][j]  # Update neuron

    @staticmethod
    def __avg_gradient_bias(dCb):
        n = len(dCb)
        m = len(dCb[0])

        avg = [[0] * len(dCb[0][j]) for j in range(m)]

        for i in range(n):
            for j in range(m):
                for l in range(len(dCb[0][j])):
                    avg[j][l] += dCb[i][j][l]

        for j in range(m):
            for l in range(len(dCb[0][j])):
                avg[j][l] /= n

        return avg

    @staticmethod
    def __avg_gradient_weight(dCw):
        n = len(dCw)
        p = len(dCw[0])

        avg = [[[0] * len(dCw[0][j][l]) for l in range(len(dCw[0][j]))] for j in range(p)]

        for i in range(n):
            for j in range(p):
                for l in range(len(dCw[0][j])):
                    for o in range(len(dCw[0][j][l])):
                        avg[j][l][o] += dCw[i][j][l][o]

        for j in range(p):
            for l in range(len(dCw[0][j])):
                for o in range(len(dCw[0][j][l])):
                    avg[j][l][o] /= n

        return avg
