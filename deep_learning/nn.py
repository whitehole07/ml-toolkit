import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from deep_learning.activation_functions import relu, drelu, sigmoid, dsigmoid

# Set the random seed
# np.random.seed(42)  # TODO: remove afterwards


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
    def __init__(self, input_size: int, layer_size: int, input_layer=False, activation_function="relu"):
        # Init neurons
        self.input_size = input_size
        self.layer_size = layer_size
        
        # Activation function
        self.sigma = relu if activation_function == "relu" else sigmoid
        self.dsigma = drelu if activation_function == "relu" else dsigmoid

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
            return z, self.sigma(z)
        else:
            return z.reshape((-1, )), z.reshape((-1, ))


class FFNN(object):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None,
                 activation_function: str = "relu"):
        # Hyper-parameters
        self.alpha = None       # Learning rate
        self.batch_size = None  # Batch size
        self.epochs = None      # Number of epochs

        # Activation function
        self.sigma = relu if activation_function == "relu" else sigmoid
        self.dsigma = drelu if activation_function == "relu" else dsigmoid

        # Train sets
        self.x_train = x_train if x_train.ndim > 1 else x_train.reshape((-1, 1))  # Size of (samples, features)
        self.y_train = y_train if y_train.ndim > 1 else y_train.reshape((-1, 1))  # Size of (samples, outputs)

        # Validation sets
        if x_val is not None and y_val is not None:
            self.x_val = x_val if x_val.ndim > 1 else x_val.reshape((-1, 1))  # Size of (samples, features)
            self.y_val = y_val if y_val.ndim > 1 else y_val.reshape((-1, 1))  # Size of (samples, outputs)
        else:
            self.x_val = None
            self.y_val = None

        # Init residual list
        self.residual_train = []
        self.residual_val = []

        # Sizes
        self.n_features = x_train.shape[1]
        self.n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        self.hidden_size = []

        # Layers
        self.layers = [Layer(self.n_features, self.n_features, input_layer=True), Layer(self.n_features, self.n_outputs)]  # Init Input Output Layer

    @property
    def MSE_train(self):
        return np.mean((1 / self.n_outputs) * (np.sum(self.residual_train[-1]**2, axis=1)))

    @property
    def MSEs_train(self):
        return [np.mean((1 / self.n_outputs) * (np.sum(residual**2, axis=1))) for residual in self.residual_train]

    @property
    def MSE_val(self):
        return np.mean((1 / self.n_outputs) * (np.sum(self.residual_val[-1] ** 2, axis=1)))

    @property
    def MSEs_val(self):
        return [np.mean((1 / self.n_outputs) * (np.sum(residual ** 2, axis=1))) for residual in self.residual_val]

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

    def train(self, *, alpha=0.001, batch_size=32, epochs=50):
        # Set hyper-parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs

        # Init losses
        self.residual_train = []
        self.residual_val = []

        # Mini-batches Stochastic Gradient Descent (mSGD)
        # Train for the specified number of epochs
        i = 1
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

            # Train: Evaluate MSE at end of epoch
            y_pred_train = self.predict(X_shuffle)
            self.residual_train.append(y_pred_train - y_shuffle)

            # Val: Evaluate MSE at end of epoch
            y_pred_val = self.predict(self.x_val)
            self.residual_val.append(y_pred_val - self.y_val)

            print("epoch", i, "train", self.MSE_train, "val", self.MSE_val)
            i += 1

    def predict(self, z_in: np.ndarray):
        # Feed Forward
        _, _, pred = self.__feed_forward(z_in)
        return pred  # Network Activations

    def print_network(self):
        """
        Print a graphical representation of a neural network given a list of layer sizes.
        """
        # print the input layer
        print("Input Layer (%d)" % self.layers[0].layer_size)
        print("       |        ")

        # print the hidden layers
        for i, layer in enumerate(self.layers[1:-1], 1):
            print("Hidden Layer #%d (%d)" % (i, layer.layer_size))
            print("       |        ")

        # print the output layer
        print("Output Layer (%d)" % self.layers[-1].layer_size)

    def plot_mse(self):
        """
        Plot the MSE (Mean Squared Error) values over epochs.

        Args:
            self: Instance of the neural network class.

        Returns:
            None
        """
        # set the style of the plot
        plt.style.use('ggplot')

        # create a figure and axes object
        fig, ax = plt.subplots()

        # plot the MSE values with a blue line and a circle marker
        ax.plot(self.MSEs_train, color='blue', marker='o', markersize=4)
        ax.plot(self.MSEs_val, color='green', marker='o', markersize=4)

        # set the x and y axis labels with a larger font size
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)

        # set the title with a larger font size
        ax.set_title('MSE Evolution over Epochs', fontsize=14)

        # add grid lines and set the line width of the grid
        ax.grid(True, linewidth=0.5)

        # remove the top and right spines of the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # set the x-axis ticks to integer values
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Tight layout
        plt.tight_layout()

        # Add legend
        plt.legend(labels=['Training', 'Validation'])

        # show the plot
        plt.show()

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
            pred = np.vstack([pred, activations[-1]]) if pred.size else activations[-1]
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
                delta_l_next.append((2 / self.n_outputs) * -(y_batch_row[j] - y_pred_batch_row[j]) * self.dsigma(zs_batch_row[-1][j]))

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
                        dl += delta_l_next[k] * self.layers[l+1].W[k, j] * self.dsigma(zs_batch_row[l][j])
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

