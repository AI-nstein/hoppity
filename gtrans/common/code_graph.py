from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
from gtrans.common.configs import cmd_args
from gtrans.common.consts import AST_EDGE_TYPE, VAR_LINK_TYPE, PREV_TOKEN_TYPE, SEPARATOR as SEP
from gtrans.common.consts import USELESS_NODES, CONTENT_NODE_TYPE

class AstNode():
    def __init__(self, node_type, value=None):
        self.node_type = node_type
        self.value = value
        self.index = None
        self.parent = None
        self.children = []

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, ch_node, pos=None):
        assert ch_node.parent is None
        if pos is None:
            self.children.append(ch_node)
        else:
            self.children.insert(pos, ch_node)
        ch_node.parent = self

    def child_rank(self, ch_node):
        for i, ch in enumerate(self.children):
            if ch.index == ch_node.index:
                return i
        return None

    def remove_child(self, ch_node):
        found = False
        for i, ch in enumerate(self.children):
            if ch.index == ch_node.index:
                found = True
                del self.children[i]
                break
        assert found

def eq_val(x, y, contents=None):
    if x is None and y == 'None':
        return True
    if x == 'None' and y is None:
        return True
    if x is None and y is None:
        return True
    return x == y or (not cmd_args.penalize_unknown and contents and (y == "UNKNOWN" or y is None or y not in contents) and x == "UNKNOWN")


def eq_types(x, y):
    if x == y:
        return True

    x_s = x.split(SEP)
    y_s = y.split(SEP)

    if not len(x_s) == len(y_s):
        return False

    assert len(x_s) == len(y_s)

    for (x2, y2) in zip(x_s, y_s):
        if not x2 == y2:
            return False

    return True

def tree_equal(x, y, contents=None, name=None):
    DEBUG = False

    if isinstance(x, AST):
        x = x.root_node
    if isinstance(y, AST):
        y = y.root_node

    y_children = copy.deepcopy(y.children)

    if not eq_types(x.node_type, y.node_type) or not eq_val(x.value, y.value, contents):
        if DEBUG:
            print("types:", x.node_type == y.node_type)
            print("values", x.value == y.value)
            print(x.node_type, y.node_type, x.value, type(x.value), y.value, type(y.value))
        return False

    if len(x.children) != len(y_children):
        if DEBUG:
            print(len(x.children), len(y_children))
            print(x.node_type, y.node_type, x.value, y.value)
            print(x.children)
            for ch in x.children:
                print(ch.node_type)
            print(y_children)
            for ch in y_children:
                print(ch.node_type)
        return False

    assert len(x.children) == len(y_children)

    for i in range(len(x.children)):
        chx = x.children[i]
        chy = y_children[i]

        if not tree_equal(chx, chy, contents, name):
            return False

    return True


class AST():
    def __init__(self):
        self.nodes = []
        self.root_node = None
        self._edits_made = []

    def append_edit(self, edit):
        if not hasattr(self, '_edits_made'):
            self._edits_made = []
        self._edits_made.append(edit)

    def get_edits(self):
        if not hasattr(self, '_edits_made'):
            return []
        return self._edits_made

    @property
    def num_nodes(self):
        return len(self.nodes)

    def add_node(self, ast_node):
        ast_node.index = self.num_nodes
        self.nodes.append(ast_node)

    def new_node(self, node_type, value):
        ast_node = AstNode(node_type=node_type, value=value)
        self.add_node(ast_node)
        return ast_node

    def remove_node(self, ast_node):
        p = ast_node.parent
        ast_node.parent = None
        assert p is not None
        p.remove_child(ast_node)

    def _relabeling(self, node):
        self.add_node(node)
        for ch in node.children:
            ch.parent = node
            self._relabeling(ch)

    def reset_index(self):
        self.nodes = []
        self._relabeling(self.root_node)


class CgNode():
    def __init__(self, index, node_type, name=None):
        self.node_type = node_type
        self.name = name
        self.ast_node = None
        self.index = index
        self.in_edge_list = []
        self.out_edge_list = []

    def add_in_edge(self, src, edge_type):
        self.in_edge_list.append((edge_type, src))

    def add_out_edge(self, dst, edge_type):
        self.out_edge_list.append((edge_type, dst))


class CodeGraph():
    def __init__(self, ast, refs, contents=None):
        self.ast = ast
        self.node_list = []
        self.edge_list = []
        self.refs = refs
        old_contents = {} if not contents else contents

        for idx, node in enumerate(ast.nodes):
            assert idx == node.index
            cg_node = self.add_node(node.node_type, node.value)
            cg_node.ast_node = node

        for node in ast.nodes:
            for ch in node.children:
                self.add_double_dir_edge(node.index, ch.index, AST_EDGE_TYPE)

        self.contents = {}
        num_ast_nodes = self.num_nodes
        for i in range(num_ast_nodes):
            node = self.node_list[i]
            if node.name is not None and node.node_type not in USELESS_NODES:
                if not node.name in self.contents:
                    self.contents[node.name] = self.add_node(CONTENT_NODE_TYPE, node.name)
                c_node = self.contents[node.name]
                assert node.index != c_node.index

                self.add_double_dir_edge(node.index, c_node.index, VAR_LINK_TYPE)

        for k in old_contents.keys():
            if k not in self.contents:
                self.contents[k] = self.add_node(CONTENT_NODE_TYPE, node.name)


        if refs is not None:
            for node in ast.nodes:
                if not str(node.index) in refs: 
                    continue
                for ref_idx in refs[str(node.index)]:
                    if int(ref_idx) >= len(self.node_list):
                        continue

                    self.add_double_dir_edge(node.index, int(ref_idx), VAR_LINK_TYPE)

        self.prev_node = None
        self.add_order_link(ast.root_node)

    @property
    def num_nodes(self):
        return len(self.node_list)

    def add_order_link(self, ast_node):
        if len(ast_node.children) == 0:  # leaf node
            if self.prev_node is not None:
                self.add_double_dir_edge(self.prev_node.index, ast_node.index, PREV_TOKEN_TYPE)
            self.prev_node = ast_node
        else:
            for ch in ast_node.children:
                self.add_order_link(ch)

    def add_node(self, node_type, node_name):
        idx = len(self.node_list)
        node = CgNode(idx, node_type, node_name)
        self.node_list.append(node)
        return node

    def add_directed_edge(self, src_idx, dst_idx, edge_type):
        x = self.node_list[src_idx]
        y = self.node_list[dst_idx]
        x.add_out_edge(y, edge_type)
        y.add_in_edge(x, edge_type)
        self.edge_list.append((src_idx, dst_idx, edge_type))

    def add_double_dir_edge(self, src_idx, dst_idx, edge_type):
        self.add_directed_edge(src_idx, dst_idx, edge_type)
        self.add_directed_edge(dst_idx, src_idx, edge_type+1)
