import numpy as np


def gaussian_kernel(distance, sigma):
    return (1/np.sqrt(2 * np.pi)) * np.exp(-(distance**2)/(2 * sigma**2))


class KernelRegression(object):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None):
        # Hyper-parameters
        self.sigma = None

        # Kernel function
        self.kernel = gaussian_kernel

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

    def predict(self, x, sigma=1):
        # Set hyperparameter
        self.sigma = sigma

        # Iterate over train dataset
        N = self.x_train.shape[0]

        # Compute denominator for each row in x
        den = np.sum(self.kernel(np.linalg.norm(self.x_train[:, np.newaxis, :] - x, axis=2), self.sigma), axis=0).reshape(-1, 1)

        # Compute weights
        W = (N * self.kernel(np.linalg.norm(self.x_train[:, np.newaxis, :] - x, axis=2), self.sigma).T)/den

        # Predict
        return np.dot(W, self.y_train)/N
