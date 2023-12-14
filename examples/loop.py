import torch

# 100 batches of size 16 of 3 by 2 tensors
data = [torch.rand(16,3,32,32) for _ in range(100)]

W = torch.rand(31,32)

total = torch.zeros(32,32)

for batch in data:
    x = W @ batch
    total = total + x



