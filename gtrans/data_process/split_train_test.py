from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import random
import json
import re
import pickle as cp
import os
import csv
import multiprocessing
from gtrans.common.configs import cmd_args
from gtrans.data_process.utils import code_group_generator, get_bug_prefix, get_source
from tqdm import tqdm


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    print(cmd_args.save_dir)
    cooked_gen = code_group_generator(cmd_args.save_dir, file_suffix=['_buggy.pkl', '_fixed.pkl', '_gedit.txt'])

    all_file_list = []
    unique_file_list = []
    file_set = set()
    for file_tuple in tqdm(cooked_gen):
        if any([not os.path.isfile(f) for f in file_tuple]):
            continue

        buggy_file, fixed_file, diff_file = file_tuple
        sample_name = get_bug_prefix(buggy_file)

        if cmd_args.raw_srcs is None:
            all_file_list.append(sample_name)
            unique_file_list.append(sample_name)
            continue

        b_src = get_source(sample_name)

        if not b_src:
            print("src for", sample_name, "does not exist")
            continue

        
        with open(b_src, "rb") as f:
            c = f.read()

        #f_label = b_src.replace("_buggy.js", "_loc_label.json")

        if not os.path.exists(diff_file):
            print("file", diff_file, "does not exist")
            continue
        with open(diff_file, "r") as f:
            l = f.read()

        h = (c, l)
        all_file_list.append(sample_name)
        if h in file_set:
            continue
        file_set.add(h)
        unique_file_list.append(sample_name)

    print("total # files", len(all_file_list), "unique #", len(unique_file_list))
    file_list = unique_file_list

    random.shuffle(file_list)
    num_test = int(len(file_list) * cmd_args.test_pct)
    num_val = int(len(file_list) * cmd_args.val_pct)
    num_train = len(file_list) - num_test - num_val

    starts = [0, num_test, num_test + num_val]
    nums = [num_test, num_val, num_train]
    suffix = ['test', 'val', 'train']

    for st, num, suff in zip(starts, nums, suffix):
        f_idx = os.path.join(cmd_args.save_dir, '%s.txt' % suff)
        with open(f_idx, 'w') as f:
            for i in range(st, st + num):
                f.write('%s\n' % file_list[i])
