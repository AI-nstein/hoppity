from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from gtrans.common.configs import cmd_args
from gtrans.common.consts import HOPPITY_HOME
from gtrans.data_process.ast_utils import build_shift_node_ast as build_ast

from tqdm import tqdm
import json
import subprocess
import os
import sys

def clean_unknown(local_vocab, global_vocab, val):
    if val in local_vocab or val in global_vocab:
        return val
    else:
        return "UNKNOWN"

def add_to_vocab(VOCAB, ident):
    if ident in VOCAB.keys():
        VOCAB[ident] += 1
    else:
        VOCAB[ident] = 1

def get_bug_prefix(buggy_file):
    fname = buggy_file.split('/')[-1]
    return '_'.join(fname.split('_')[:-1])

def code_group_generator(data_root, file_suffix=['_buggy.json', '_buggy.js', '_fixed.json', '_ast_diff.txt']):
    files = os.listdir(data_root)

    for fname in files:
        abs_path = os.path.join(data_root, fname)

        if os.path.isdir(abs_path):
            for t in code_group_generator(abs_path, file_suffix):
                yield t
        elif fname.endswith(file_suffix[0]):
            prefix = fname.split(file_suffix[0])[0]
            local_names = []
            for suff in file_suffix:
                if suff == "_buggy.js":
                    my_prefix = prefix.replace("SHIFT_", "")
                else:
                    my_prefix = prefix
                local_names.append(os.path.join(data_root, my_prefix + suff))
            yield tuple(local_names)

def get_ref_edges(src_file, pkl_file):
    try:
        script = os.path.join(HOPPITY_HOME, "gtrans", "data_process", "find_all_refs.js")
        s = subprocess.check_output(["node", script, src_file, pkl_file])
        s = s.decode()
    except (subprocess.CalledProcessError, AttributeError) as e:
        print(e)
        print("FIND ALL REFS FAILED")
        print(src_file, pkl_file)
        sys.exit(1)
         
    try:
        return json.loads(s)
    except json.decoder.JSONDecodeError as e:
        print("JSON ERROR", s)
        sys.exit(1)

def dict_depth(d, depth=0):
    if isinstance(d, dict):
        return max(dict_depth(v, depth+1) for k, v in d.items())
    elif isinstance(d, list) and len(d) > 0:
        return max(dict_depth(item, depth+1) for item in d)
    return depth

def get_source(sample_name, buggy=True):
    suffix = "_buggy" if buggy else "_fixed"
    babel_suffix = suffix + "_babel.js"

    for raw_src in cmd_args.raw_srcs:
        b_src1 = os.path.join(raw_src, sample_name.replace("_SHIFT","") + suffix + ".js")
        b_src2 = os.path.join(raw_src, sample_name.replace("_SHIFT","") + babel_suffix)
        b_src3 = os.path.join(raw_src, sample_name + suffix + ".js")

        if not os.path.exists(b_src1) and not os.path.exists(b_src2) and not os.path.exists(b_src3):
            continue

        b_src = b_src1 if os.path.exists(b_src1) else b_src2
        b_src = b_src3 if not os.path.exists(b_src) else b_src

        return b_src

    return None

