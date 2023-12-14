import ast
from functools import reduce
import sys
from datatypes import *
from helpers import *
import numpy as np
import copy

import_aliases = {}  # maps imports to their user-defined aliases,
# ex. import numpy as np -> name : numpy, alias : np

################################################################################
############################   STATMENT LISTS   ################################
################################################################################

# take in possible contexts and modify
def parse_stmt_list(stmts, contexts: List[Context]) -> List[Context]:

    # possible_contexts = [context]
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
                # print("Import Aliases -", import_aliases)  # For debugging
        elif isinstance(elt, ast.ImportFrom):
            for item in elt.names:
                name = item.name
                alias = item.asname
                wholename = elt.module + "." + name
                if alias == None:
                    import_aliases[name] = wholename
                else:
                    import_aliases[alias] = wholename
                # print("Import Aliases -", import_aliases)  # For debugging

        # Assign!
        elif isinstance(elt, ast.Assign):
            # We are assigning variables to a tensor,
            # add them to the context
            for context in contexts:
                t = typecheck_expr(elt.value, context)
                if (
                    isinstance(t, Tensor)
                    or isinstance(t, Dataloader)
                    or isinstance(t, Function)
                ):
                    name = (elt.targets[0]).id
                    context[name] = t

        # Expression
        elif isinstance(elt, ast.Expr):
            for context in contexts:
                typecheck_expr(elt.value, context)

        # If statement
        elif isinstance(elt, ast.If):
            # typecheck the test
            for context in contexts:
                # this should probably raise an exception or something
                test = typecheck_expr(elt.test, context)
            # Typecheck the body
            contexts1 = parse_stmt_list(elt.body, copy.deepcopy(contexts))  # Deep copy?
            # Typecheck the else
            contexts2 = parse_stmt_list(elt.orelse, copy.deepcopy(contexts))
            contexts = contexts1 + contexts2

        elif isinstance(elt, ast.For):
            if isinstance(elt.iter, ast.Name):
                for context in contexts:
                    iterable = context[elt.iter.id]
                    if isinstance(iterable, Dataloader):
                        get_types_iter(elt.target, iterable, context)
                    else:
                        continue
            contexts = parse_stmt_list(elt.body, contexts)

        elif isinstance(elt, ast.FunctionDef):
            # create new tensor args for all args w type annotations
            # add arguments to context
            # typecheck the body
            args = [arg.arg for arg in elt.args.args]
            name = elt.name
            body = elt.body
            for context in contexts:
                context[name] = Function(name, body, args)
        elif isinstance(elt, ast.Return):
            return_expr = elt.value
            for context in contexts:
                context["$return_type"] = (
                    None
                    if return_expr is None
                    else typecheck_expr(return_expr, context)
                )

    return contexts


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
            if not tensors_compatable(left, right):
                return

            # Add/Subtract tensors
            if isinstance(expr.op, ast.Add) or isinstance(expr.op, ast.Sub):
                return typecheck_add_sub(left, right, expr.lineno)

            # Elementwise Mult Tensors
            elif isinstance(expr.op, ast.Mult):
                return typecheck_mult(left, right, expr.lineno)

            # Matrix Multiplication
            elif isinstance(expr.op, ast.MatMult):
                return typecheck_matmul(left, right, expr.lineno)

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

    elif isinstance(expr, ast.ListComp):
        # if the expression inside the list comp is a static tensor, create dataloader for it.
        elem_type = typecheck_expr(expr.elt, context)
        # print(elem_type)
        if isinstance(elem_type, Tensor):
            return Dataloader([elem_type])
        elif isinstance(elem_type, list):
            return Dataloader(elem_type)

    elif isinstance(expr, ast.Tuple):
        tuple_types = []
        for elt in expr.elts:
            elem_type = typecheck_expr(elt, context)
            tuple_types.append(elem_type)
        return tuple_types

    # other cases
    else:
        return context


################################################################################
############################   FUNCTION CALLS   ################################
################################################################################


def typecheck_function_call(expr, context):
    func = expr.func

    # return a list of types corresponding to the different return types of the function,
    # assume the function just has one return type for now
    # r type is gonna be bound to the last element in the context
    if isinstance(func, ast.Name):
        if func.id in context.keys():
            # add arguments to context
            defn = context[func.id]
            if not isinstance(defn, Function):
                print("call id not a function on line", expr.lineno)
                return

            func_context = copy.deepcopy(context)
            for arg, named_arg in zip(expr.args, defn.args):
                arg_t = typecheck_expr(arg, context)

                if (
                    isinstance(arg_t, Tensor)
                    or isinstance(arg_t, Dataloader)
                    or isinstance(arg_t, Function)
                ):
                    func_context[named_arg] = arg_t
            body_ctxs = parse_stmt_list(defn.body, [func_context])
            return_types = []
            for ctx in body_ctxs:
                # print(ctx.keys())
                if not ("$return_type" in ctx.keys()):
                    # print("not in")
                    return_types.append(None)
                else:
                    return_types.append(ctx["$return_type"])

            for r_type in return_types:
                if r_type != return_types[0]:
                    print(
                        "warning, function returns different types on line", expr.lineno
                    )

            return return_types[0]

        # builtin
        else:
            return

    elif isinstance(func, ast.Attribute):
        body = typecheck_expr(expr.func.value, context)

        if func.attr == "multiply" or func.attr == "mul":
            t1 = typecheck_expr(expr.args[0], context)
            t2 = typecheck_expr(expr.args[1], context)
            return typecheck_mult(t1, t2, expr.lineno)

        elif func.attr == "add" or func.attr == "subtract":
            t1 = typecheck_expr(expr.args[0], context)
            t2 = typecheck_expr(expr.args[1], context)
            return typecheck_add_sub(t1, t2, expr.lineno)

        elif (
            func.attr == "square"
            or func.attr == "sqrt"
            or func.attr == "power"
            or func.attr == "pow"
        ):
            return typecheck_expr(expr.args[0], context)

        elif (
            func.attr == "stack"
            or func.attr == "concatenate"
            or func.attr == "cat"
            or func.attr == "exp"
        ):
            dtype, device = parse_keywords(expr)
            axis = None
            if func.attr == "stack":
                axis = 0
            size = []
            for kwarg in expr.keywords:  # Determine axis
                if kwarg.arg == "axis":
                    axis = kwarg.value.value
            for arg in expr.args:  # Determine elements
                if hasattr(arg, "elts"):
                    for elt in arg.elts:
                        t = typecheck_expr(elt, context)
                        if func.attr == "stack":
                            if size == []:
                                size = t.size
                            elif t.size != size:
                                print(
                                    f"cannot stack tensors of differing sizes! Sizes {size} and {t.size} on line",
                                    expr.lineno,
                                )
                                return
                        else:
                            tsize = t.size
                            if axis == None:
                                tsize = [np.prod(tsize)]  # Flatten
                            else:
                                # Sizes along everything besides axis match up!
                                if size == []:
                                    size = tsize
                                if (
                                    size[0:axis] == tsize[0:axis]
                                    and size[axis + 1 :] == tsize[axis + 1 :]
                                ):
                                    size[axis] += tsize[axis]
            if func.attr == "stack":
                if axis != None and (axis >= len(size) or axis < 0):
                    print("Axis out of bounds", axis, "on line", expr.lineno)
                    return
                new_size = size[0:axis] + [1] + size[axis:]
            else:
                new_size = size
            t = Tensor(
                size=new_size,
                type=import_aliases[expr.func.value.id],
                data_type=dtype,
                device=device,
            )
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
                    print(
                        f"Invalid reshape! Cannot reshape tensor of size {body.size} to tensor of size {new_size}",
                        expr.lineno,
                    )
                    return
                t = Tensor(
                    size=new_size,
                    type=body.type,
                    device=body.device,
                    data_type=body.data_type,
                )
                return t
            else:
                print("Cannot call reshape on non-tensor", expr.lineno)
                return

        # Handles torch.tensor calls
        elif func.attr == "tensor" or func.attr == "array":
            size = obtain_manual_size(expr.args)
            dtype, device = parse_keywords(expr)
            if func.attr == "tensor":
                t_t = "torch"
            else:
                t_t = "array"
            t = Tensor(size=size, type=t_t, data_type=dtype, device=device)
            return t

        # Handles np.arrange
        if func.attr == "arange":
            args = [arg.value for arg in expr.args]
            if len(args) == 1:
                size = [args[0]]  # arange(stop)
            elif len(args) == 2:
                size = [args[1] - args[0]]  # arange(start, stop)
            elif len(args) == 3:
                size = [int((args[1] - args[0]) / args[2])]  # arange(start, stop, step)
            dtype, device = parse_keywords(expr)
            t = Tensor(size=size, type="numpy", data_type=dtype, device=device)
            return t

        if func.attr == "tile":
            t_type = typecheck_expr(expr.args[0], context)
            reps = expr.args[1]
            if isinstance(t_type, Tensor):
                t_size = t_type.size
                if hasattr(reps, "elts"):
                    d_dim = len(reps.elts)
                    # add additional 1 dims
                    if d_dim > len(t_size):
                        t_size = [1] * (d_dim - len(t_size)) + t_size
                    reps_size = []
                    for rep in reps.elts:
                        reps_size.append(rep)
                    res_size = tile_dims(t_size, reps_size)
                    t = Tensor(
                        size=res_size,
                        type="numpy",
                        data_type=t_type.data_type,
                        device=t_type.device,
                    )
                    return t
            else:
                print("argument for tile must be a tensor type", expr.lineno)
                return


# Obtain the AST from the Python Script
file_path = "examples/python.py" if len(sys.argv) < 2 else sys.argv[1]
with open(file_path, "r") as file:
    file_content = file.read()
ast_ = ast.parse(file_content)

# Parse!
contexts = parse_stmt_list(ast_.body, [Context()])
# for context in contexts:
#     print("the context is ")
#     print(context)

