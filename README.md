## Setup / Installation
Install yojson (for JSON reading)
```
brew install ocaml opam
opam install yojson
```
## Running
1. Populate the file ```python.py``` with the Python code you would like to have typechecked

2. In the terminal at the root of this directory, run ```python3 generate_ast.py```. This will create a JSON file containing the AST of your python program.

3. Compile and run the OCaml typechecker:

    Compile OCaml Code:
    ```
    ocamlfind ocamlc -o ast_reader -package yojson -linkpkg ast_reader.ml
    ```
    Run the OCaml Program:
    ```
    ./ast_reader
    ```

## Resources
Info about the OCaml JSON parser we are using:
https://mjambon.github.io/mjambon2016/yojson-doc/Yojson.Basic.html

Info about Python AST:
https://docs.python.org/3/library/ast.html