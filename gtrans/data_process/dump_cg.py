from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from gtrans.common.configs import cmd_args
from gtrans.common.consts import DEVICE
from gtrans.common.dataset import Dataset
from gtrans.model.gtrans_model import GraphTrans
from gtrans.common.code_graph import tree_equal

def get_save_dir(part):
    out_dir = os.path.join(cmd_args.save_dir, 'part-%d' % part)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return out_dir

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.set_num_threads(1)
    torch.manual_seed(cmd_args.seed)
    torch.autograd.set_detect_anomaly(True)

    vocab_name = 'vocab_%s.npy' % cmd_args.vocab_type
    print('loading value vocab from', vocab_name)
    const_val_vocab = np.load(os.path.join(cmd_args.data_root, vocab_name), allow_pickle=True).item()
    Dataset.set_value_vocab(const_val_vocab)
    Dataset.add_value2vocab(None)
    Dataset.add_value2vocab("UNKNOWN")
    print('global value table size', Dataset.num_const_values())

    dataset = Dataset(cmd_args.data_root, cmd_args.gnn_type, 
                      data_in_mem=cmd_args.data_in_mem,
                      resampling=cmd_args.resampling)

    f_per_part = 1000
    cur_part = 0
    cnt = 0
    cur_out_dir = get_save_dir(cur_part)
    for s in tqdm(dataset.data_samples):
        for fname, cg in [(s.f_bug, s.buggy_code_graph), (s.f_fixed, s.fixed_code_graph)]:
                    out_fname = '.'.join(fname.split('.')[:-1]) + '.json'
                    out_fname = os.path.basename(out_fname)
                    out_fname = os.path.join(cur_out_dir, out_fname)
                    d = {}
                    d['nodes'] = []
                    for node in cg.node_list:
                        name = 'None' if node.name is None else node.name
                        d['nodes'].append((node.index, node.node_type, name))
                    d['edges'] = []
                    for e in cg.edge_list:
                        d['edges'].append(e)
                    with open(out_fname, 'w') as fout:
                        json.dump(d, fout)
        cnt += 1
        if cnt >= f_per_part:
            cnt = 0
            cur_part += 1
            cur_out_dir = get_save_dir(cur_part)
