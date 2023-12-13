import numpy as np

x = np.random.rand(2,3,2,5)
y = np.tile(x, (2, 3, 4, 5))
print(y.shape)
