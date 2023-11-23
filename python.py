import torch
import numpy as np

x = torch.rand(5, 3, dtype=torch.float32, device="cuda")
y = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float32, device="cuda")
z = np.arange(15, dtype=np.int64)
p = np.arrange(15).reshape(3, 5)
res = x + y
print(res)