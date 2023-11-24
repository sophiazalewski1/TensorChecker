import ast

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

class Context(dict):
    def __str__(self):
        items = [f"'{key}': {value if isinstance(value, TensorType) else value}" for key, value in self.items()]
        return "\nContext - {" + ",\n           ".join(items) + "}"

import_aliases = {}
context = Context()

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
                if(alias == None):
                    import_aliases[name] = name
                else:
                    import_aliases[name] = alias
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
            print("expression!")
            pass
        
        else:
            print("other")

# Typechecks an expression
def typecheck_expr(expr):

    # Function call - check if it's a builtin numpy or torch function
    if isinstance(expr, ast.Call):

        if(hasattr(expr, "func")):
            return typecheck_function_call(expr)
    
    else:
        print("expr not call",expr,expr.lineno)

def typecheck_function_call(expr):
    if(not hasattr(expr, "func")): 
        print(expr.lineno, "no func", expr)
        return
    func = expr.func
    if(not hasattr(func, "value")): 
        print(expr.lineno, "no val")
        return 

    # Function acts on a value
    # ex. x.reshape(...)
    if(hasattr(func.value, "func")):
        print("HEREEE")
        t = typecheck_function_call(func.value)
        print(t)
    
    # Function is a direct call
    # ex. torch.tensor(...)
    elif(hasattr(func.value, "id")):

        ###################### PYTORCH BUILTIN #####################
        if(func.value.id == import_aliases["torch"]):

            # torch.rand
            if(func.attr == "rand"):
                size = [arg.value for arg in expr.args]
                dtype, device = parse_keywords(expr)
                t = TensorType(size = size, type = "torch", data_type = dtype, device=device)
                return t

            # torch.tensor
            elif(func.attr == "tensor"):
                size = obtain_manual_size(expr.args)
                dtype, device = parse_keywords(expr)
                t = TensorType(size = size, type = "torch", data_type = dtype, device=device)
                return t

        ####################### NUMPY BUILTIN #####################
        if(func.value.id == import_aliases["numpy"]):
            
            # np.arrange
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

# CHECK THIS FUNCTION!!!
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

# Obtain the AST from the Python Script
file_path = 'python.py' 
with open(file_path, 'r') as file:
    file_content = file.read()
ast_ = ast.parse(file_content)

# Parse!
parse_stmt_list(ast_.body)