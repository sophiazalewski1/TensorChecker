import torch
x = torch.rand(5, 3)
y = torch.tensor([[1., -1.], [1., -1.]])
z = x + y
print(z)