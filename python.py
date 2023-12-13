import torch
import numpy as np
import numpy.random as r2

x = torch.rand(5,1)
y = torch.rand(1,5)

#a = np.stack([x,x],axis=0)
#b = np.concatenate([x,x,x,x],axis=1)
z = np.multiply(x,x)

#z = x @ y