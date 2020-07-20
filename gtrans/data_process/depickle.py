import pickle as cp
import sys
import json

def unpickle(fname):
    with open(fname, "rb") as f:
        ast = cp.load(f)
    return ast

def ast_to_dict(ast):
    obj = {}
    obj["children"] = []
    for child in ast.children:
        obj["children"].append(ast_to_dict(child))

    obj["node_type"] = ast.node_type
    obj["index"] = ast.index
    obj["value"] = ast.value

    return obj

print(json.dumps(ast_to_dict(unpickle(sys.argv[1]).root_node)))
