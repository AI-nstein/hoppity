from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import sys
from gtrans.common.code_graph import AST, AstNode
from gtrans.common.consts import SEPARATOR

sys.setrecursionlimit(1500)

def create_child(node, ast):
        src = ast.new_node(node.node_type, node.value)
        if len(node.children) == 0:
            return src

        count = 0
        for child in node.children:
            src.add_child(create_child(child, ast), count)
            count += 1

        return src

def build_shift_node_ast_from_json(json_node, parent_node, ast):
    if isinstance(json_node, dict):
        assert 'type' in json_node
        value = str(json_node['value']) if 'value' in json_node else None
        ast_node = ast.new_node(node_type=json_node['type'], value=value)
        for key in json_node:
            if key == 'type' or key == 'value':
                continue
            if isinstance(json_node[key], dict):                
                ch_node = build_shift_node_ast_from_json(json_node[key], ast_node, ast)
                ch_node.node_type = key + SEPARATOR + ch_node.node_type
            elif isinstance(json_node[key], list):
                ch_node = ast.new_node(key, value=None)
                build_shift_node_ast_from_json(json_node[key], ch_node, ast)
            else:
                value = None if json_node[key] is None else str(json_node[key])
                ch_node = ast.new_node(key, value=value)
            ast_node.add_child(ch_node)
        return ast_node
    elif isinstance(json_node, list):
        assert parent_node is not None
        for d in json_node:
            if isinstance(d, dict):
                ch_node = build_shift_node_ast_from_json(d, None, ast)
            else:
                assert d is None
                ch_node = ast.new_node('None', value=None)
            parent_node.add_child(ch_node)
    else:
        raise NotImplementedError



def build_shift_node_ast(fname):
    with open(fname, 'r') as f:
        try:
            root_node = json.load(f)
        except (json.decoder.JSONDecodeError, RecursionError) as e:
            return None

        ast = AST()
        root = build_shift_node_ast_from_json(root_node, None, ast)
        ast.root_node = root
    return ast

