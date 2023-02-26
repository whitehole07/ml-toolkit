import numpy as np
from matplotlib import pyplot as plt

from examples.wine_quality.dataset import red_wine_dataset
from regression.kernel_regression import KernelRegression

# set the style of the plot
plt.style.use('ggplot')

# Get dataset
data, X_train, Y_train, X_val, Y_val = red_wine_dataset()

# Init Regression
k = KernelRegression(X_train, Y_train, X_val, Y_val)

# Predict for validation
Y_pred_val = k.predict(X_val, sigma=0.05)

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
