from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
from copy import deepcopy
from gtrans.common.code_graph import tree_equal
from gtrans.common.consts import SEPARATOR, OP_DEL_NODE, OP_REPLACE_VAL, OP_REPLACE_TYPE, OP_ADD_NODE
from gtrans.data_process.utils import dict_depth
from gtrans.model.utils import adjust_attr_order, get_attr_idx
from gtrans.data_process.ast_utils import create_child

class ShiftNodeGraphEditParser():
    def __init__(self, ast_src, ast_dst, diff_file):
        self.ast_src = ast_src
        self.ast_dst = ast_dst
        self.diff_file = diff_file
        with open(diff_file, 'r') as f:
            edits = json.load(f)
            self.diff_logs = edits


    def locate_node(self, node, path, pos):
        if pos == len(path):
            return node
        if path[pos] == 'type' or path[pos] == 'value':
            assert pos + 1 == len(path)
            return node
        if path[pos].isdigit():
            idx = int(path[pos])
            return self.locate_node(node.children[idx], path, pos + 1)

        for ch in node.children:
            if ch.node_type.split(SEPARATOR)[0] == path[pos]:
                return self.locate_node(ch, path, pos + 1)
        return None


    def parse_edits(self):
        ast_src = deepcopy(self.ast_src)
        
        graph_edits = []
        src_val_set = set()

        for node in ast_src.nodes:
            if node.value is not None:
                src_val_set.add(node.value)

        error_log = None        
        for edit in self.diff_logs:            
            assert 'op' in edit and 'path' in edit

            if edit["op"] == "add" and dict_depth(edit["value"]) > 1:
                error_log = 'unsopported add subtree'
                break


            path = edit['path'].strip().split('/')
            if edit["op"] == "add" and not path[-1] == "value":
                ast_node = self.locate_node(self.ast_dst.root_node, path, 1)
                ast_par_node = self.locate_node(ast_src.root_node, path[0:-1], 1)

                if len(ast_node.children) > 0:
                    error_log = "unsupported add - translates to a tree in our internal AST" 
                    break
                
            else:
                ast_node = self.locate_node(ast_src.root_node, path, 1)

            if ast_node is None:
                print(path, self.diff_file)
            assert ast_node is not None

            if edit['op'] == 'remove':
                if path[-1] == "value":
                    ast_node.value = None
                    graph_edits.append(SEPARATOR.join([OP_REPLACE_VAL, str(ast_node.index), str(None)]))
                else:
                    ast_src.remove_node(ast_node)
                    graph_edits.append(SEPARATOR.join([OP_DEL_NODE, str(ast_node.index)]))
            elif edit['op'] == 'add':
                if path[-1] == "value":
                    ast_node.value = str(edit["value"])
                    graph_edits.append(SEPARATOR.join([OP_REPLACE_VAL, str(ast_node.index), str(ast_node.value)]))
                else:
                    ch = create_child(ast_node, ast_src)

                    if path[-1].isdigit():
                        ast_par_node.add_child(ch, int(path[-1]))
                    else:
                        ch_idx = get_attr_idx(ast_par_node.node_type, ch.node_type)
                        ast_par_node.add_child(ch, ch_idx)

                    sib_num = int(path[-1]) if path[-1].isdigit() else ch_idx
                    node_details = SEPARATOR.join([OP_ADD_NODE, str(ast_par_node.index), str(sib_num), ast_node.node_type, str(ast_node.value)])
                    graph_edits.append(node_details)

            elif edit['op'] == 'replace':
                assert 'value' in edit

                if isinstance(edit['value'], (bool, float, int, str)):
                    val = str(edit['value'])
                else:
                    error_log = 'unsupported replace type: ' + str(type(edit['value']))
                    break

                e_type = OP_REPLACE_TYPE if path[-1] == 'type' else OP_REPLACE_VAL
                if e_type == OP_REPLACE_TYPE:
                    new_type = self.locate_node(self.ast_dst.root_node, path, 1).node_type
                    ast_node.node_type = new_type
                    ast_node.children = adjust_attr_order(new_type, ast_node.children)
                else:
                    ast_node.value = val
                
                graph_edits.append(SEPARATOR.join([e_type, str(ast_node.index), val]))
            else:
                raise NotImplementedError

        if error_log is None:
            assert tree_equal(ast_src.root_node, self.ast_dst.root_node)

        return error_log, graph_edits
