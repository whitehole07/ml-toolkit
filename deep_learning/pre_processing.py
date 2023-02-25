import numpy as np


# Convert labels to one-hot encoding
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    oh_encode = np.zeros((n_labels, n_unique_labels))
    oh_encode[np.arange(n_labels), labels] = 1
    return oh_encode
