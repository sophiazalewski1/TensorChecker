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
    def __init__(self, size: size, type: str, data_type=None, device=None):
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

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        elif self.type != other.type:
            return False
        elif self.data_type != other.data_type:
            return False
        elif self.size != other.size:
            return False
        elif self.device != other.device:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


# Stores function name
class Function(Type):
    def __init__(self, name, body=None, args=None):
        self.body = body
        self.args = args
        self.name = name

    def __str__(self):
        return ",".join([arg for arg in self.args])

    def __eq__(self, other):
        if not isinstance(other, Function):
            return False
        return self.name == other.name and len(self.args) == len(other.args)

    def __ne__(self, other):
        return not self.__eq__(other)


# stores
class Dataloader(Type):
    def __init__(self, tensor_ts: List[Tensor]):
        self.tensor_ts = tensor_ts

    def __str__(self):
        return "Dataloader of " + str(self.tensor_ts)

    def __eq__(self, other):
        for tensor_t, other_t in zip(self.tensor_ts, other.tensor_ts):
            if tensor_t != other_t:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)


# Maps names of type Tensor to their correspinding Tensor info
class Context(dict):
    def __str__(self):
        items = [f"'{key}': {value}" for key, value in self.items()]
        return "\nContext - {" + ",\n           ".join(items) + "}"
