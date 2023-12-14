import numpy as np
import random


def multiply2(A, B):
    return A @ B


x = np.random.rand(3, 2)
y = np.random.rand(2, 3)

# error types of A and B should match up
B = multiply2(x, y)

