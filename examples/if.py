import torch

condition = True
condition2 = False
condition3 = True
x = torch.rand(3,2)

if condition:
    x = torch.rand(3,2)
    y = torch.rand(2,3)
elif condition2:
    if condition3:
        x = torch.rand(6,6)
        y = torch.rand(6,6)
    else:
        x = torch.rand(8,2)
        y = torch.rand(2,8)
else:
    x = torch.rand(2,3)
    y = torch.rand(3,2)
z = y @ x