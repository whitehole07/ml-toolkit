import numpy as np
from matplotlib import pyplot as plt

from deep_learning.nn import FFNN
from examples.wine_quality.dataset import red_wine_dataset

# Get dataset
data, X_train, Y_train, X_val, Y_val = red_wine_dataset()

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
