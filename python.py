import torch
import numpy as np

x = torch.rand(5, 3, dtype=torch.float32)
y = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float32)
p = np.arange(15).reshape(3, 5)
x.reshape(1,4)
w = np.arange(0,5,0.5,dtype=int)
print(p)
res = x + y
print(res)