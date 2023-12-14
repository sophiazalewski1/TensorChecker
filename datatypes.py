from typing import List
size = List[int]

class TensorError(Exception):
    pass

# Can be either a Tensor or a Function
class Type:
    def __init__():
        pass

# Stores size, type, dtype, device
class Tensor(Type):
    def __init__(self, size : size, type : str, data_type=None, device=None):
        self.size = size
        self.type = type
        self.device = device
        self.data_type = data_type
        if data_type == None:  
            if type == "numpy":
                self.data_type = "float64"
            elif type == "torch":
                self.data_type = "float32"
        if device is None:
            self.device == "cpu"

    def __str__(self):
        return f'(size={self.size}, type="{self.type}", dtype="{self.data_type}", device="{self.device}")'

# Stores function name 
class Function(Type):
    def __init__(self, name : str):
        self.name = name
        # inputs to the function
        self.inputs = None
        # list of constraints to be enforced on the input shapes if they are tensors
        self.constraints = None
        # return type is some function of the inputs
        self.return_type = None
    def __str__(self):
        return self.name

# stores 
class Dataloader(Type):
    def __init__(self, tensor_ts : List[Tensor]):
        self.tensor_ts = tensor_ts
    def get_tensor_type(self):
        return self.tensor_ts
    def __str__(self):
        return "Dataloader of " + str(self.tensor_ts)

# Maps names of type Tensor to their correspinding Tensor info
class Context(dict):
    def __str__(self):
        items = [
            f"'{key}': {value if isinstance(value, Tensor) else value}"
            for key, value in self.items()
        ]
        return "\nContext - {" + ",\n           ".join(items) + "}"

class ContextWithArgs(Type):
    def __init__(self, args):
        self.Context = Context()
        for arg in args:
            # use None as a placeholder and fill in constraints as you go
            Context[arg] = Tensor(-1, "Any", "Any")
        # constraints contains the arguments of the function and which dims have to match up
        self.constraints = []

MNIST_T = Tensor((1,32,32), "torch")

MNIST_DATALOADER = Dataloader(MNIST_T)