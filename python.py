import torch
import numpy as np
import numpy.random as r2

x = torch.rand(4,5)
y = x.reshape(5,4)
z = np.random.rand(4,5,device="cuda").reshape(20,1)