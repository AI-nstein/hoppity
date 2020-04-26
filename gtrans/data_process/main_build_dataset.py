from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import csv
import json
import multiprocessing
import os
import random
import pickle as cp
import numpy as np
import torch
from tqdm import tqdm
from gtrans.common.configs import cmd_args
from gtrans.common.consts import js_keywords
from gtrans.data_process import build_ast, GraphEditParser
from gtrans.data_process.utils import code_group_generator, get_bug_prefix, get_ref_edges


def make_graph_edits(file_tuple):
    vocab = {}

    if any([not os.path.isfile(f) for f in file_tuple]):
        return file_tuple, (None, None), None, ('buggy/fixed/ast_diff file missing', None), vocab

    f_bug, f_bug_src, f_fixed, f_diff = file_tuple

    sample_name = get_bug_prefix(f_bug)

    if os.path.exists(os.path.join(cmd_args.save_dir, sample_name + "_refs.npy")):
        return file_tuple, (None, None), None, ('Already exists', None), vocab
    elif f_bug in processed:
        return file_tuple, (None, None), None, ('Already processed', None), vocab

    ast_bug = build_ast(f_bug)
    ast_fixed = build_ast(f_fixed)
    if not ast_bug or not ast_fixed or ast_bug.num_nodes > cmd_args.max_ast_nodes or ast_fixed.num_nodes > cmd_args.max_ast_nodes:
        return file_tuple, (None, None), None, ('too many nodes in ast', None), vocab

    all_nodes = ast_bug.nodes + ast_fixed.nodes
    for node in all_nodes:
        if node.value and not node.value in js_keywords:
            if node.value in vocab.keys():
                vocab[node.value] += 1
            else:
                vocab[node.value] = 1

    gp = GraphEditParser(ast_bug, ast_fixed, f_diff)

    buggy_pkl = os.path.join(cmd_args.save_dir, '%s_buggy.pkl' % sample_name)
    with open(buggy_pkl, 'wb') as f:
        cp.dump(ast_bug, f)

    fixed_pkl = os.path.join(cmd_args.save_dir, '%s_fixed.pkl' % sample_name)
    with open(fixed_pkl, 'wb') as f:
        cp.dump(ast_fixed, f)

    buggy_refs = get_ref_edges(f_bug_src, buggy_pkl)

    return file_tuple, (ast_bug, ast_fixed), buggy_refs, gp.parse_edits(), vocab


if __name__ == '__main__':

    if not os.path.exists("processed.txt"):
        open("processed.txt", "w").close()

    with open("processed.txt", "r") as f:
        processed = f.read()

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    raw_folder = cmd_args.data_root

    file_gen = code_group_generator(raw_folder)
    print("new vocab")
    VOCAB = {}
    
    pool = multiprocessing.Pool(cmd_args.num_cores)

    sample_list = []
    pbar = tqdm(pool.imap_unordered(make_graph_edits, file_gen))
    
    f_error_log = open(cmd_args.save_dir + '/error_log.csv', 'w')
    f_test_log = os.path.join(cmd_args.save_dir, "test.txt")
    f_train_log = os.path.join(cmd_args.save_dir, "train.txt")  

    with open(f_test_log, "w") as f:
        f.write("")

    with open(f_train_log, "w") as f:
        f.write("")

    writer = csv.writer(f_error_log)

    for file_tuple, (ast_bug, ast_fixed), ref_edges, (error_log, edits), vocab in pbar:
        buggy_file, buggy_src, fixed_file, diff_file = file_tuple

        with open("processed.txt", "a") as f:
            f.write(buggy_file + "\n")

        for k, v in vocab.items():
            if k not in VOCAB.keys():
                VOCAB[k] = v
            else:
                VOCAB[k] += v

        sample_name = get_bug_prefix(buggy_file)

        if error_log is not None:
            writer.writerow([sample_name, error_log])
        else:
            sample_list.append(sample_name)
            pbar.set_description('# valid: %d' % len(sample_list))
            with open(os.path.join(cmd_args.save_dir, '%s_gedit.txt' % sample_name), 'w') as f:
                json_arr = []
                for row in edits:
                    edit_obj = {}
                    edit_obj["edit"] = row
                    json_arr.append(edit_obj)

                json.dump(json_arr, f, indent=4)

            ref_out = os.path.join(cmd_args.save_dir, '%s_refs.npy' % sample_name)
            np.save(ref_out, ref_edges)

            
    f_error_log.close()
    print(len(sample_list), 'files loaded')

    for i, sample_name in enumerate(sample_list):
        sample_name = sample_list[i]
        idx = random.randint(1, 10)
        
        test = (idx == 10)
        if test:
            file_to_write = f_test_log
        else:
            file_to_write = f_train_log

        with open(file_to_write, "a") as f:
            f.write(sample_name + "\n")


    vocab_out = os.path.join(cmd_args.save_dir, "vocab.npy")
    np.save(vocab_out, VOCAB)
