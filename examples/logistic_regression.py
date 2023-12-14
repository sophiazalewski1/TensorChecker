# logistic regression example adapted from this notebook:

import math
import numpy as np


def sigmoid(z: float) -> float:
    """A numerically stable implementation of sigmoid."""
    # Calculate the log of the sum of the exponetials of 0 and -z.
    x = np.logaddexp(0, -z)
    # Return the exponential of -x.
    return np.exp(-x)


def grad(y: float, y_hat: float, x_j: float) -> float:
    return (y_hat - y) * x_j


# Define the features.
x = np.array([1, 3, 2])
# Initialize the weights to 0.
theta = np.array([0, 0, 0, 0])
# Set the learning rate to 0.1
eta = 0.1
# Set the true value to 0.
y = 1


for iteration in range(2, 30):
    print(f"\nITERATION {iteration}")
    gradient = np.array(
        (grad(y, sigmoid(theta @ x), x[0])),
        (grad(y, sigmoid(theta @ x), x[1])),
        (grad(y, sigmoid(theta @ x), x[2])),
    )
    print(f"gradient = {gradient}")
    theta = np.subtract(eta * gradient, theta)
    print(f"theta = {theta}")

