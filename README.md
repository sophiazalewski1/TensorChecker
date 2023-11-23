Install yojson (for JSON reading)
```
brew install ocaml opam
opam install yojson
```
Compile OCaml Code:
```
ocamlfind ocamlc -o ast_reader -package yojson -linkpkg ast_reader.ml
```
Run the OCaml Program:
```
./ast_reader
```
Info about JSON parsing:
https://mjambon.github.io/mjambon2016/yojson-doc/Yojson.Basic.html

Info about Python AST: