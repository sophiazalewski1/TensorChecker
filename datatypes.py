from typing import List
size = List[int]

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
        if data_type == None:  # Not sure if these defaults are correct
            if type == "numpy":
                self.data_type = "float64"
            elif type == "torch":
                self.data_type = "float32"
        if device == None:
            self.device == "cpu"

    def __str__(self):
        return f'(size={self.size}, type="{self.type}", dtype="{self.data_type}", device="{self.device}")'

# Stores function name 
class Function(Type):
    def __init__(self, name : str):
        self.name = name
    def __str__(self):
        return self.name

# Maps names of type Tensor to their correspinding Tensor info
class Context(dict):
    def __str__(self):
        items = [
            f"'{key}': {value if isinstance(value, Tensor) else value}"
            for key, value in self.items()
        ]
        return "\nContext - {" + ",\n           ".join(items) + "}"