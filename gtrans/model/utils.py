from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
import random
import numpy as np
import torch
from gtrans.common.consts import SEPARATOR, attr_order
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AutoregModel(nn.Module):
    def __init__(self):
        super(AutoregModel, self).__init__()
        self.rand_flag = False

    def update_beam_stats(self, new_ops, new_ll, beam_size, prefix_sum=None):
        device = new_ops.device
        bsize = len(self.sample_indices)
        if isinstance(new_ops, torch.Tensor):
            new_ops = new_ops.cpu().data.numpy()
        if isinstance(new_ll, torch.Tensor):
            new_ll = new_ll.cpu().data.numpy()

        if isinstance(self.ll, np.ndarray) or isinstance(self.ll, torch.Tensor):
            fn_ll = lambda i: self.ll[i]
        else:
            fn_ll = lambda i: 0

        new_sample_ids = []
        old_sample_ids = []
        total_new = 0
        joint_ll = []
        list_op_ids = []

        node_offsets = []
        offset = 0
        for graph in self.sample_buf:
            node_offsets.append(offset)
            offset += graph.pg.num_nodes

        node_ids = []
        for i in range(bsize):
            list_options = []
            for j in self.sample_indices[i]:
                prev_ll = fn_ll(j)
                if prefix_sum is not None:
                    num_choices = prefix_sum[0] if i == 0 else prefix_sum[i] - prefix_sum[i - 1]
                    num_choices = num_choices.item()
                    op_lim = min(new_ops.shape[1], num_choices)
                else:
                    op_lim = new_ops.shape[1]
                for k in range(op_lim):
                    op_tuple = (prev_ll + new_ll[j, k], new_ops[j, k], j)
                    list_options.append(op_tuple)
            if self.rand_flag:
                random.shuffle(list_options)
            else:
                list_options.sort(key=lambda x: -x[0])

            list_options = list_options[:beam_size]
            for x in list_options:
                joint_ll.append(x[0])
                list_op_ids.append(x[1])
                old_sample_ids.append(x[2])
                offset = node_offsets[x[2]]
                node_ids += list(range(offset, offset + self.sample_buf[x[2]].pg.num_nodes))
            new_sample_ids.append(range(total_new, total_new + len(list_options)))
            total_new += len(list_options)

        self.sample_indices = new_sample_ids
        self.sample_buf = [self.sample_buf[i] for i in old_sample_ids]
        self.ll = np.array(joint_ll, dtype=np.float32)
        if self.states is not None:
            idx = torch.LongTensor(old_sample_ids).to(device)
            self.states = self.states[idx]
        self.node_embedding = self.node_embedding[node_ids]
        list_op_ids = torch.LongTensor(list_op_ids).to(device)

        self.cur_edits = [self.cur_edits[i] for i in old_sample_ids]
        return list_op_ids, old_sample_ids

def adjust_refs(refs, idx):
    new_refs = {}
    for k, v in refs.items():
        if not k == str(idx):
            new_refs[k] = []
            for item in v:
                if not str(idx) == item:
                    new_refs[k].append(item)
    return new_refs

def select_child(attr, children, select_func):
    for child in children:
        if select_func(child, attr):
            children.remove(child)
            return copy.deepcopy(child)


def get_attr_idx(par_type, attr):
    if SEPARATOR in par_type:
        par_type = par_type.split(SEPARATOR)[1]

    if SEPARATOR in attr:
        attr = attr.split(SEPARATOR)[0]

    if not par_type in attr_order:
        return -1

    return attr_order[par_type].index(attr)


def adjust_attr_order(new_type, children):
    if SEPARATOR in new_type:
        new_type = new_type.split(SEPARATOR)[1]

    if not new_type in attr_order or len(children) == 0:
        return children

    attrs = attr_order[new_type]
    out_children = []
    for attr in attrs:
        ch = select_child(attr, children, lambda node, attr: attr in node.node_type)
        if ch:
            out_children.append(ch)
    for node in children:
        node.parent = None
    return out_children

neg_logits = -10000000 + 100

def get_randk(ll_all, num_tries):
    indices = []
    step_ll = []
    np_ll = ll_all.cpu().data.numpy()
    for i in range(ll_all.shape[0]):
        x_p = []
        for j in range(ll_all.shape[1]):
            if np_ll[i, j] > neg_logits:
                x_p.append(j)
        random.shuffle(x_p)
        li = x_p[:num_tries]
        ls = []
        for j in li:
            ls.append(np_ll[i, j])
        step_ll.append(ls)
        indices.append(li)
    step_ll = torch.FloatTensor(step_ll).to(ll_all.device)
    indices = torch.LongTensor(indices).to(ll_all.device)
    return step_ll, indices

def get_rand_one(ll_all):
    np_ll = ll_all.cpu().data.numpy()
    indices = []
    for i in range(ll_all.shape[0]):
        x_p = []
        for j in range(ll_all.shape[1]):
            if np_ll[i, j] > neg_logits:
                x_p.append(j)
        indices.append(np.random.choice(x_p))
    indices = torch.LongTensor(indices).to(ll_all.device)
    return indices
