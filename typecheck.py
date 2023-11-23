import ast

import_aliases = {}

class TensorType:
    def __init__(self, size, data_type, type, device="cpu"):
        self.size = size
        self.device = device
        self.data_type = data_type
        self.type = type # numpy or torch

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
                print("Aliases -", import_aliases) # For debugging

        # Assign!
        elif isinstance(elt, ast.Assign): 
            typecheck_expr(elt.value)

        # Expression
        elif isinstance(elt, ast.Expr): 
            pass

# Typechecks an expression
def typecheck_expr(expr):

    # Function call
    if isinstance(expr, ast.Call):
        typecheck_function_call(expr.func)

def typecheck_function_call(func):
    if(not hasattr(func, "value")): return 

    # Function acts on another function
    if(hasattr(func.value, "func")):
        typecheck_function_call(func.value.func)
    
    # If we are directly calling a function
    elif(hasattr(func.value, "id")):

        ###################### PYTORCH BUILTIN #####################
        if(func.value.id == import_aliases["torch"]):

            # torch.rand
            if(func.attr == "rand"):
                print("torch.rand call!")

            # torch.tensor
            elif(func.attr == "tensor"):
                print("torch.tensor call!")

        ####################### NUMPY BUILTIN #####################
        if(func.value.id == import_aliases["numpy"]):
            print("numpy fnnnn")


# Obtain the AST from the Python Script
file_path = 'python.py' 
with open(file_path, 'r') as file:
    file_content = file.read()
ast_ = ast.parse(file_content)

# Parse!
parse_stmt_list(ast_.body)