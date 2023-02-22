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
