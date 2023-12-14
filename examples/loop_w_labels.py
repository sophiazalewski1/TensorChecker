import torch

# 100 batches of size 16 of 3 by 2 tensors
data = [(1, torch.rand(16,3,32,32)) for _ in range(100)]

W = torch.rand(31,32)

total = torch.zeros(32,32)

for inputs, labels in data:
    x = W @ inputs
    x = inputs - labels
    total = total + x