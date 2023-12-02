import ast
from functools import reduce

############################## DATA TYPES ################################

# Stores info about a tensor (size, type, dtype, device)
class TensorType:
    def __init__(self, size, type, data_type=None, device=None):
        self.size = size
        self.type = type 
        self.device = device
        self.data_type = data_type
        if(data_type == None): # Not sure if these defaults are correct
            if(type == "numpy"): self.data_type = "float64"
            elif(type == "torch"): self.data_type = "float32"
        if(device == None):
            self.device == "cpu"
    def __str__(self):
        return (f"(size={self.size}, type=\"{self.type}\", dtype=\"{self.data_type}\", device=\"{self.device}\")")
class FunctionType:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

# Maps names to their correspinding tensor info
class Context(dict):
    def __str__(self):
        items = [f"'{key}': {value if isinstance(value, TensorType) else value}" for key, value in self.items()]
        return "\nContext - {" + ",\n           ".join(items) + "}"

import_aliases = {}
context = Context()

############################## TYPECHECKING ################################

# Parse a stmt list
def parse_stmt_list(stmts):
    for elt in stmts:

        ###################### PARSE IMPORTS #####################
        """build up a dictionary of (import_name : alias) values so that
        when we see a function call, we can identify if it is a builtin
        function from an imported module. For example, the code:

            import numpy as np 
            np.arrange(...) 
            
        can be recognized as defining a numpy tensor because we map the 
        alias 'np' to the import 'numpy' """
        ###########################################################

        if isinstance(elt, ast.Import):
            for item in elt.names:
                name = item.name
                alias = item.asname
                if(alias == None): import_aliases[name] = name
                else: import_aliases[alias] = name
                print("Import Aliases -", import_aliases) # For debugging
        elif isinstance(elt, ast.ImportFrom):
            for item in elt.names:
                name = item.name
                alias = item.asname
                wholename = elt.module + "." + name
                if(alias == None): import_aliases[name] = wholename
                else: import_aliases[alias] = wholename
                print("Import Aliases -", import_aliases) # For debugging

        # Assign!
        elif isinstance(elt, ast.Assign): 
            
            # We are assigning variables to a tensor, 
            # add them to the context

            t = typecheck_expr(elt.value)
            if isinstance(t, TensorType):
                for targ in elt.targets:
                    context[targ.id] = t
                    print(context)

        # Expression
        elif isinstance(elt, ast.Expr): 
            print("expression")
            pass

        else:
            print("other")

################################ HELPERS #################################

# Checks if tensors are on the same device and have same dtype
def tensors_compatable(tensor1, tensor2):
    if not isinstance(tensor1, TensorType):
        return True # Any operation between non-tensors is compatable by default
    if not isinstance(tensor2, TensorType):
        return True # Any operation between non-tensors is compatable by default
    res = True
    if(tensor1.type != tensor2.type):
        print("Types of tensors do not match!")
        res = False
    if(tensor1.device != tensor2.device):
        print("Devices of tensors do not match!")
        res = False
    return res

# CHECK THIS FUNCTION!!!
# Obtains the size from function calls with explit tensor value declarations
# ex. torch.tensor([[-1,-1],[2,4]])
def obtain_manual_size(args):
    def helper(elts):
        size = []
        num_elts = len(elts)
        for arg in elts:
            if(hasattr(arg, "elts")):
                res = helper(arg.elts)
                res.append(num_elts)
                return res
        return [num_elts]
    for arg in args:
        if(hasattr(arg,"elts")):
            return helper(arg.elts)
        
# Obtains tensor info (dtype and device) from function call args
def parse_keywords(expr):
    data_type = None
    device = None
    for keyword in expr.keywords: 
        if(keyword.arg == "dtype"):
            if (hasattr(keyword.value, "attr")):
                data_type = keyword.value.attr
            elif (hasattr(keyword.value, "id")):
                data_type = keyword.value.id
        elif(keyword.arg == "device"):
            device = keyword.value.value
    return data_type, device

##########################################################################

# Typechecks an expression
def typecheck_expr(expr):

    if isinstance(expr, ast.BinOp):
            left = typecheck_expr(expr.left)
            right = typecheck_expr(expr.right)

            # Dealing with at least one tensor
            if(isinstance(left, TensorType) or isinstance(right, TensorType)):
                
                # Check if types and devices are compatible
                if not tensors_compatable(left,right):
                    return
                
                # Add/Subtract tensors
                if(isinstance(expr.op, ast.Add) or isinstance(expr.op, ast.Sub)):
                    
                    # Adding tensors of same exact size
                    if(isinstance(left, TensorType) and 
                       isinstance(right, TensorType) and 
                       left.size == right.size):
                        return left
                    
                    # Scalar Addition
                    elif(isinstance(left, TensorType) and left.size == [1]): return right
                    elif(isinstance(left, TensorType) and not isinstance(right, TensorType)): return left
                    elif(isinstance(right, TensorType) and right.size == [1]): return left
                    elif(isinstance(right, TensorType) and not isinstance(left, TensorType)): return right

                    # ADD OTHER CASES
                    else:
                        print("Mismatch!")
                        return

                # Elementwise Mult Tensors
                elif(isinstance(expr.op, ast.Mult)):

                    # Mult tensor of compatible sizes (nxm) x (mxn)
                    if(isinstance(left, TensorType) and 
                       isinstance(right, TensorType) and 
                       left.size == right.size):
                        return left

                    # Scalar Multiplication
                    elif(isinstance(left, TensorType) and left.size == [1]): return right
                    elif(isinstance(left, TensorType) and not isinstance(right, TensorType)): return left
                    elif(isinstance(right, TensorType) and right.size == [1]): return left
                    elif(isinstance(right, TensorType) and not isinstance(left, TensorType)): return right

                    # ADD OTHER CASES
                    else:
                        print("Mismatch!")
                        return

                # Matrix Multiplication   
                elif(isinstance(expr.op, ast.MatMult)):
                    pass

    # Function call 
    elif isinstance(expr, ast.Call):
        if(hasattr(expr, "func")):
            return typecheck_function_call(expr)
    
    # Attribute
    elif isinstance(expr, ast.Attribute):
        body = typecheck_expr(expr.value)
        if(isinstance(body, FunctionType)):
            name = body.name + "." + expr.attr
            if name in import_aliases: name = import_aliases[name]
            return FunctionType(name)

    elif isinstance(expr, ast.Name):
        # name is a variable, lookup its type
        # ex. x.reshape
        name = expr.id
        if name in context: return context[name]

        # name is a bultin, ex. "torch", "numpy"
        # ex. TORCH.tesnor
        if name in import_aliases: 
            name = import_aliases[name]
            if(hasattr(expr, "attr")):
                name = name + "." + expr.attr
                if name in import_aliases: 
                    return FunctionType(import_aliases[name])
        return FunctionType(name)

    else:
        print("expr not call",expr,expr.lineno)


def typecheck_function_call(expr):

    func = expr.func
    body = typecheck_expr(expr.func.value)

    # Handles torch.rand, numpy.random.rand calls
    if(func.attr == "rand"):
        size = [arg.value for arg in expr.args]
        dtype, device = parse_keywords(expr)
        if (body.name == "torch"):
            type = "torch"
        elif (body.name == "numpy.random"):
            type = "numpy"
        t = TensorType(size = size, type = type, data_type = dtype, device=device)
        return t

    elif(func.attr == "reshape"):
        new_size = [arg.value for arg in expr.args]
        if(isinstance(body, TensorType)):
            elts1 = reduce((lambda x, y: x * y), body.size)
            elts2 = reduce((lambda x, y: x * y), new_size)
            if(elts1 != elts2):
                print("Invalid reshape!")
                return
            t = TensorType(size=new_size, type=body.type, device=body.device, data_type=body.data_type)
            return t
        else:
            print("Cannot call reshape on non-tensor")
            return

    # Handles torch.tensor calls
    elif(func.attr == "tensor"):
        size = obtain_manual_size(expr.args)
        dtype, device = parse_keywords(expr)
        t = TensorType(size = size, type = "torch", data_type = dtype, device=device)
        return t

    # Handles np.arrange
    if(func.attr == "arange"):
        args = [arg.value for arg in expr.args]
        if(len(args) == 1):
            size = args[0] # arange(stop)
        elif(len(args) == 2):
            size = args[1] - args[0] #arange(start, stop)
        elif(len(args) == 3):
            size = int((args[1] - args[0])/args[2]) # arange(start, stop, step)
        dtype, device = parse_keywords(expr)
        t = TensorType(size = size, type = "numpy", data_type = dtype, device=device)
        return t

# Obtain the AST from the Python Script
file_path = 'python.py' 
with open(file_path, 'r') as file:
    file_content = file.read()
ast_ = ast.parse(file_content)

# Parse!
parse_stmt_list(ast_.body)