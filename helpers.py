from datatypes import *

############################# PARSING AST INFO #################################

# Obtains the size from function calls with explit tensor value declarations
# ex. torch.tensor([[-1,-1],[2,4]])
def obtain_manual_size(args):
    def helper(elts):
        size = []
        num_elts = len(elts)
        for arg in elts:
            if hasattr(arg, "elts"):
                res = helper(arg.elts)
                res.append(num_elts)
                return res
        return [num_elts]

    for arg in args:
        if hasattr(arg, "elts"):
            return helper(arg.elts)

# Obtains tensor info (dtype and device) from function call args
def parse_keywords(expr):
    data_type = None
    device = None
    for keyword in expr.keywords:
        if keyword.arg == "dtype":
            if hasattr(keyword.value, "attr"):
                data_type = keyword.value.attr
            elif hasattr(keyword.value, "id"):
                data_type = keyword.value.id
        elif keyword.arg == "device":
            device = keyword.value.value
    return data_type, device


################# TYPE (Torch vs. Numpy) + DEVICE CHECKING #####################

# Checks if tensors are on the same device and have same dtype
def tensors_compatable(tensor1 : Type, tensor2 : Type) -> bool:

    # Any operation between non-tensors is compatable by default
    if not isinstance(tensor1, Tensor): return True  
    if not isinstance(tensor2, Tensor): return True 
    if tensor1.type != tensor2.type:
        print("Types of tensors do not match!")
        return False
    if tensor1.device != tensor2.device:
        print("Devices of tensors do not match!")
        return False
    return True

############################### SIZE CHECKING ##################################

# implements rules defined in matmul.py
def check_size_matmul(size1 : size, size2 : size) -> bool:
    # inner sizes match
    if size1[-1] != size2[-2]:
        print("Matmul: dimensions mismatch")
        return
    res_dims = [size1[-2], size2[-1]]
    size1 = size1[:-2]
    size2 = size2[:-2]
    # check if batch dimensions broadcast
    # automatically broadcast extra dims
    batch_dims = []
    if len(size1) < len(size2):
        batch_dims = size2[: len(size2) - len(size1)]
        size2 = size2[len(size2) - len(size1) :]
    elif len(size2) < len(size1):
        batch_dims = size1[: len(size1) - len(size2)]
        size1 = size2[len(size1) - len(size2) :]

    for (n1, n2) in zip(size1, size2):
        if n1 == 1:
            batch_dims.append(n2)
        elif n2 == 1:
            batch_dims.append(n1)
        else:
            # TODO make this message better by getting index of dims that are mismatching
            print("Matmul: cannot broadcast nonsingleton dimension")
            return
    res_dims = batch_dims + res_dims
    return res_dims

################################ FUNCTION CALLS ################################

def typecheck_add_sub(left : Type, right : Type) -> Type:
    # Adding tensors of same exact size
    if (
        isinstance(left, Tensor)
        and isinstance(right, Tensor)
        and left.size == right.size
    ):
        return left

    # Scalar Addition
    elif isinstance(left, Tensor) and left.size == [1]:
        return right
    elif isinstance(left, Tensor) and not isinstance(right, Tensor):
        return left
    elif isinstance(right, Tensor) and right.size == [1]:
        return left
    elif isinstance(right, Tensor) and not isinstance(left, Tensor):
        return right

    # ADD OTHER CASES
    else:
        print("Mismatch!")
        return
                
# Elementwise multiplication check
def typecheck_mult(left : Type, right : Type) -> Type:
    # Mult tensor of compatible sizes (nxm) x (mxn)
    if (
        isinstance(left, Tensor)
        and isinstance(right, Tensor)
        and left.size == right.size
    ):
        return left

    # Scalar Multiplication
    elif isinstance(left, Tensor) and left.size == [1]:
        return right
    elif isinstance(left, Tensor) and not isinstance(right, Tensor):
        return left
    elif isinstance(right, Tensor) and right.size == [1]:
        return left
    elif isinstance(right, Tensor) and not isinstance(left, Tensor):
        return right

    # ADD OTHER CASES
    else:
        print("Mismatch!")
        return
                
# Matrix wise multiplication check
def typecheck_matmul(left : Type, right : Type) -> Type:

    # Both tensors
    if isinstance(left, Tensor) and isinstance(right, Tensor):

        # check tensors are on the same device
        remove_left = False
        remove_right = False

        # If left/right are 1 dimensional, add dimension, flag for removal
        dims_left = left.size
        dims_right = right.size
        if len(dims_left) == 1:
            dims_left = [1] + left.size
            remove_left = True
        elif len(dims_right) == 1:
            dims_right = right.size + [1]
            remove_right = True
            print(dims_right)
        res_dim = check_size_matmul(dims_left, dims_right)

        if res_dim is None:
            return
        # cleanup added dims
        if remove_left:
            # remove the second to last dimension
            res_dim = res_dim[:-2] + res_dim[-1:]
        if remove_right:
            res_dim = res_dim[:-1]
        t_type = Tensor(res_dim, left.type, left.data_type, left.device)
        return t_type
    else:
        print("both left and right side of matrix mul have to be tensor types")
        return

