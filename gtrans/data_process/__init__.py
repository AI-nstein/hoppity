from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from gtrans.common.configs import cmd_args


if cmd_args.ast_fmt == 'gumtree':
    from gtrans.data_process.ast_utils import build_gumtree_ast as build_ast
    from gtrans.data_process.gumtree_interpreter import GumTreeGraphEditParser as GraphEditParser
elif cmd_args.ast_fmt == 'shift_node':
    from gtrans.data_process.ast_utils import build_shift_node_ast as build_ast
    from gtrans.data_process.shift_interpreter import ShiftNodeGraphEditParser as GraphEditParser
elif cmd_args.ast_fmt == 'min_node':
    from gtrans.data_process.ast_utils import build_min_node_ast as build_ast
    from gtrans.data_process.min_interpreter import MinNodeGraphEditParser as GraphEditParser


