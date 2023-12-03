""" 
matrix multiplication rules:
https://pytorch.org/docs/stable/generated/torch.matmul.html

1. If both tensors are 1-dimensional, the dot product (scalar) is returned.
        x = rand(3)
        y = rand(3)

2. If both arguments are 2-dimensional, the matrix-matrix product is returned.
        X = (3,2)
        Y = (2,3)

3. if the first argument is 1-dimensional and the second argument is 2-dimensional, 
    a 1 is prepended to its dimension for the purpose of the matrix multiply. After
    the matrix multiply, the prepended dimension is removed.
        X = rand(3)
        y = rand(3,2)
        y @ X ====> shape(2,)

4. If the first argument is 2-dimensional and the second argument is 1-dimensional, 
    the matrix-vector product is returned.
        X = rand(3,2)
        y = rand(2)

5. If both arguments are at least 1-dimensional and at least one argument is 
    N-dimensional (where N > 2), then a batched matrix multiply is returned. If the 
    first argument is 1-dimensional, a 1 is prepended to its dimension for the 
    purpose of the batched matrix multiply and removed after. If the second argument 
    is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched 
    matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are 
    broadcasted (and thus must be broadcastable). For example, 
    
    if input is a (j,1,n,n) tensor and other is a (k,n,n) tensor, out will be a 
    (j,k,n,n) tensor.
    Note that the broadcasting logic only looks at the batch dimensions when 
    determining if the inputs are broadcastable, and not the matrix dimensions. 
    For example, if input is a (j,1,n,m) tensor and other is a (k,m,p) tensor, 
    these inputs are valid for broadcasting even though the final two dimensions 
    (i.e. the matrix dimensions) are different. out will be a (j,k,n,p) tensor.

        this does not work (dimension 1 mismatch)
        x = torch.rand(8,7,2,3)
        y = torch.rand(5,3,3,2)
        x @ y

        this also does not work (dimension 0 mismatch)
        x = torch.rand(8,1,2,3)
        y = torch.rand(5,3,3,2)
        x @ y
        
        this also works, dimensions 0,1 of x are singleton (1)
        dimensions 2 and 3 are multiplied together
        size (5,3,2,2)
        x = torch.rand(1,1,2,3)
        y = torch.rand(5,3,3,2)
        x @ y

        this also works
        size (5,3,2,2)
        x = torch.rand(5, 1, 2, 3)
        y = torch.rand(1, 3, 3, 2)
        x @ y

        # doesnt typecheck (last two dimensions dont match up)
        x = torch.rand(4, 1, 3, 3)
        y = torch.rand(1, 3, 2, 2)
        x @ y

    tldr; the matrix dimensions in the last two slots are the traditional 
    "matrix multiplications" dimension, while everything before are the batch 
    dimensions are broadcastable i.e. copy-able 
"""

import torch

x = torch.rand(4, 1, 3, 3)
y = torch.rand(3)

z = x @ y
print(z.shape) # shape 4,4,1,3,2
