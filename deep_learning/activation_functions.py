import numpy as np


def relu(z):
    """
    Computes the Rectified Linear Unit (ReLU) activation function for a given input.

    Arguments:
    z -- numpy array of any shape

    Returns:
    A -- output of ReLU(z), same shape as z
    """
    A = np.maximum(0, z)
    return A


def drelu(z):
    """
    Computes the derivative of the Rectified Linear Unit (ReLU) activation function for a given input.

    Arguments:
    z -- numpy array of any shape

    Returns:
    A -- output of derivative of ReLU(z), same shape as z
    """
    return np.greater(z, 0).astype(int)


def sigmoid(z):
    """
    Compute the sigmoid of z.

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def dsigmoid(z):
    """
    Compute the derivative of the sigmoid function with respect to z.

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    ds -- The derivative of the sigmoid function with respect to z.
    """
    s = sigmoid(z)
    ds = s * (1 - s)
    return ds
