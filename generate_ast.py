import ast
import json
import sys

def ast_to_dict(node):
    if isinstance(node, ast.AST):
        fields = {k: ast_to_dict(v) for k, v in ast.iter_fields(node)}
        fields['node_type'] = node.__class__.__name__
        return fields
    elif isinstance(node, list):
        return [ast_to_dict(v) for v in node]
    else:
        return node

file_path = "examples/python.py" if len(sys.argv) < 2 else sys.argv[1]
output_file_path = 'AST.json'

with open(file_path, 'r') as file:
    file_content = file.read()

parsed_ast = ast.parse(file_content)

with open(output_file_path, 'w') as output_file:
    json.dump(ast_to_dict(parsed_ast), output_file, indent=4)
