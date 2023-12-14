import torch

# 100 batches of size 16 of 3 by 2 tensors
data = torch.rand(3, 16, 32, 32)

W = torch.rand(31, 32)

total = torch.zeros(32, 32)

for _ in range(100):
    x = W @ data
    total = total + x
