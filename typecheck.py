import ast
from functools import reduce
import sys
from datatypes import *
from helpers import *
import numpy as np

import_aliases = {} # maps imports to their user-defined aliases, 
                     # ex. import numpy as np -> name : numpy, alias : np

################################################################################
############################   STATMENT LISTS   ################################
################################################################################

def parse_stmt_list(stmts, context):

    possible_contexts = [context]
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
                if alias == None:
                    import_aliases[name] = name
                else:
                    import_aliases[alias] = name
                print("Import Aliases -", import_aliases)  # For debugging
        elif isinstance(elt, ast.ImportFrom):
            for item in elt.names:
                name = item.name
                alias = item.asname
                wholename = elt.module + "." + name
                if alias == None:
                    import_aliases[name] = wholename
                else:
                    import_aliases[alias] = wholename
                print("Import Aliases -", import_aliases)  # For debugging

        # Assign!
        elif isinstance(elt, ast.Assign):

            # We are assigning variables to a tensor,
            # add them to the context

            t = typecheck_expr(elt.value, context)
            if isinstance(t, Tensor):
                for targ in elt.targets:
                    context[targ.id] = t
                    print(context)

        # Expression
        elif isinstance(elt, ast.Expr):
            print("expression")
            pass
        
        # If statement
        elif isinstance(elt, ast.If):
            # Typecheck the body
            context1 = parse_stmt_list(context.copy()) # Deep copy?

            # Typecheck the else
            context2 = parse_stmt_list(context.copy())

        else:
            print("other")

################################################################################
##############################   EXPRESSIONS   #################################
################################################################################

# Typechecks an expression for a given context
def typecheck_expr(expr, context):

    if isinstance(expr, ast.BinOp):
        left = typecheck_expr(expr.left, context)
        right = typecheck_expr(expr.right, context)

        # Dealing with at least one tensor, same type and device
        if isinstance(left, Tensor) or isinstance(right, Tensor):
            if not tensors_compatable(left, right): return

            # Add/Subtract tensors
            if isinstance(expr.op, ast.Add) or isinstance(expr.op, ast.Sub):
                return typecheck_add_sub(left, right)

            # Elementwise Mult Tensors
            elif isinstance(expr.op, ast.Mult):
                return typecheck_mult(left, right)

            # Matrix Multiplication
            elif isinstance(expr.op, ast.MatMult):
                return typecheck_matmul(left, right)

    # Function call
    elif isinstance(expr, ast.Call):
        if hasattr(expr, "func"):
            return typecheck_function_call(expr, context)

    # Attribute
    elif isinstance(expr, ast.Attribute):
        body = typecheck_expr(expr.value, context)
        if isinstance(body, Function):
            name = body.name + "." + expr.attr
            if name in import_aliases:
                name = import_aliases[name]
            return Function(name)

    elif isinstance(expr, ast.Name):
        # name is a variable, lookup its type
        # ex. x.reshape
        name = expr.id
        if name in context:
            return context[name]

        # name is a bultin, ex. "torch", "numpy"
        # ex. TORCH.tesnor
        if name in import_aliases:
            name = import_aliases[name]
            if hasattr(expr, "attr"):
                name = name + "." + expr.attr
                if name in import_aliases:
                    return Function(import_aliases[name])
        return Function(name)

    else:
        print("expr not call", expr, expr.lineno)

################################################################################
############################   FUNCTION CALLS   ################################
################################################################################

def typecheck_function_call(expr, context):

    func = expr.func
    body = typecheck_expr(expr.func.value, context)

    if (func.attr == "multiply"):
        t1 = typecheck_expr(expr.args[0], context)
        t2 = typecheck_expr(expr.args[1], context)
        return typecheck_mult(t1, t2)
    
    elif (func.attr == "add" or func.attr == "subtract"):
        t1 = typecheck_expr(expr.args[0], context)
        t2 = typecheck_expr(expr.args[1], context)
        return typecheck_add_sub(t1, t2)
    
    elif (func.attr == "sqrt" or func.attr == "power"):
        return typecheck_expr(expr.args[0], context)

    elif (func.attr == "stack" or func.attr == "concatenate" or func.attr == "cat"):
        dtype, device = parse_keywords(expr)
        axis = None
        if(func.attr == "stack"): 
            axis = 0
        size = []
        for kwarg in expr.keywords: # Determine axis
            if kwarg.arg == "axis":
                axis = kwarg.value.value
        for arg in expr.args: # Determine elements
            if hasattr(arg, "elts"):
                for elt in arg.elts:
                    t = typecheck_expr(elt, context)
                    if func.attr == "stack":
                        if (size == []):
                            size = t.size
                        elif (t.size != size):
                            print("cannot stack tensors of differing sizes!")
                            return
                    else:
                        tsize = t.size
                        if axis == None:
                            tsize = [np.prod(tsize)] # Flatten 
                        else:
                            # Sizes along everything besides axis match up!
                            if (size == []):
                                size = tsize
                            if (size[0:axis] == tsize[0:axis] and size[axis+1:] == tsize[axis+1:]):
                                size[axis] += tsize[axis]
        if(func.attr == "stack"):
            print("here",size)
            if axis != None and (axis >= len(size) or axis < 0):
                print("axis out of bounds", axis)
                return
            new_size = size[0:axis] + [1] + size[axis:]
        else:
            new_size = size
        t = Tensor(size=new_size, type=import_aliases[expr.func.value.id], data_type=dtype, device=device)
        return t

    # Handles torch.rand, numpy.random.rand calls
    elif func.attr == "rand":
        size = [arg.value for arg in expr.args]
        dtype, device = parse_keywords(expr)
        if body.name == "torch":
            type = "torch"
        elif body.name == "numpy.random":
            type = "numpy"
        t = Tensor(size=size, type=type, data_type=dtype, device=device)
        return t

    elif func.attr == "reshape":
        new_size = [arg.value for arg in expr.args]
        if isinstance(body, Tensor):
            elts1 = reduce((lambda x, y: x * y), body.size)
            elts2 = reduce((lambda x, y: x * y), new_size)
            if elts1 != elts2:
                print("Invalid reshape!")
                return
            t = Tensor(
                size=new_size,
                type=body.type,
                device=body.device,
                data_type=body.data_type,
            )
            return t
        else:
            print("Cannot call reshape on non-tensor")
            return

    # Handles torch.tensor calls
    elif func.attr == "tensor":
        size = obtain_manual_size(expr.args)
        dtype, device = parse_keywords(expr)
        t = Tensor(size=size, type="torch", data_type=dtype, device=device)
        return t

    # Handles np.arrange
    if func.attr == "arange":
        args = [arg.value for arg in expr.args]
        if len(args) == 1:
            size = args[0]  # arange(stop)
        elif len(args) == 2:
            size = args[1] - args[0]  # arange(start, stop)
        elif len(args) == 3:
            size = int((args[1] - args[0]) / args[2])  # arange(start, stop, step)
        dtype, device = parse_keywords(expr)
        t = Tensor(size=size, type="numpy", data_type=dtype, device=device)
        return t

# Obtain the AST from the Python Script
file_path = "python.py" if len(sys.argv) < 2 else sys.argv[1]
with open(file_path, "r") as file:
    file_content = file.read()
ast_ = ast.parse(file_content)

# Parse!
parse_stmt_list(ast_.body, Context())