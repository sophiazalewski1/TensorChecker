open Yojson.Basic
open Yojson.Basic.Util

type json = Yojson.Basic.t

(* Program body *)
let rec traverse_ast (ast : json) =
    match ast with
    | `Assoc(("body",`List(json_elts))::_) -> let () = print_endline "body" in traverse_elts json_elts
    | _ -> let () = print_endline "no body with list in AST" in ()

(* Traverse a list of json elements *)
and traverse_elts (json_elts : json list) =
    match json_elts with
    | [] -> let () = print_endline "done traversing body" in ()
    | x::xs -> (
        let () = traverse_elt x in
        traverse_elts xs
    )
(* Typecheck an individual element *)
and traverse_elt (elt : json) =
  match elt with

  (* Assign! *)
  | `Assoc(("targets",x)::("value",y)::_::
           ("node_type",`String("Assign"))::[]) -> (
    let () = print_endline "found!" in ()
  ) 

  (* Value *)
  | `Bool(_) -> let () = print_endline "bool" in ()
  | `Float(_) -> let () = print_endline "float" in ()
  | `Int(_) -> let () = print_endline "int" in ()
  | `Null -> let () = print_endline "null" in ()

  | `List [x] -> let () = print_endline "yessir" in ()

  | _ -> let () = print_endline "bruh" in ()

(* Parse the JSON generated from our Python script *)
let json_content = Yojson.Basic.from_file "AST.json"
let ast = traverse_ast json_content

