from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter_max as orig_smax
from torch_scatter import scatter_min as orig_smin

def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return orig_smax(src, index, dim, out, dim_size, fill_value)[0]

def scatter_min(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return orig_smin(src, index, dim, out, dim_size, fill_value)[0]

def get_agg(agg_type):
    if agg_type == 'sum':
        return scatter_add
    elif agg_type == 'mean':
        return scatter_mean
    elif agg_type == 'max':
        return scatter_max
    elif agg_type == 'min':
        return scatter_min
    else:
        raise NotImplementedError


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}

class ReadoutNet(nn.Module):
    def __init__(self, node_state_dim, output_dim, max_lv, act_func, out_method, readout_agg, act_last, bn):
        super(ReadoutNet, self).__init__()

        self.out_method = out_method
        self.max_lv = max_lv
        self.readout_agg = get_agg(readout_agg)
        self.act_last = act_last
        self.act_func = NONLINEARITIES[act_func]
        self.readout_funcs = []
        self.bn = bn
        if output_dim is None:
            self.embed_dim = node_state_dim
            for i in range(self.max_lv + 1):
                self.readout_funcs.append(lambda x: x)
        else:
            self.embed_dim = output_dim
            for i in range(self.max_lv + 1):
                self.readout_funcs.append(nn.Linear(node_state_dim, output_dim))
                if self.out_method == 'last':
                    break
            self.readout_funcs = nn.ModuleList(self.readout_funcs)

        if self.out_method == 'gru':
            self.final_cell = nn.GRUCell(self.embed_dim, self.embed_dim)
        if self.bn:
            out_bn = [nn.BatchNorm1d(self.embed_dim) for _ in range(self.max_lv + 1)]
            self.out_bn = nn.ModuleList(out_bn)

    def forward(self, list_node_states, g_idx, num_graphs):
        assert len(list_node_states) == self.max_lv + 1
        if self.out_method == 'last':
            out_states = self.readout_funcs[0](list_node_states[-1])
            if self.act_last:
                out_states = self.act_func(out_states)
            graph_embed = self.readout_agg(out_states, g_idx, dim=0, dim_size=num_graphs)
            return graph_embed, (g_idx, out_states)

        list_node_embed = [self.readout_funcs[i](list_node_states[i]) for i in range(self.max_lv + 1)]
        if self.act_last:
            list_node_embed = [self.act_func(e) for e in list_node_embed]
        if self.bn:
            list_node_embed = [self.out_bn[i](e) for i, e in enumerate(list_node_embed)]
        list_graph_embed = [self.readout_agg(e, g_idx, dim=0, dim_size=num_graphs) for e in list_node_embed]
        
        if self.out_method == 'gru':
            out_embed = list_graph_embed[0]
            out_node_embed = list_node_embed[0]    
            for i in range(1, self.max_lv + 1):            
                out_embed = self.final_cell(list_graph_embed[i], out_embed)
                out_node_embed = self.final_cell(list_node_embed[i], out_node_embed)
        else:
            if self.out_method == 'sum':
                fn = torch.sum
            elif self.out_method == 'mean':
                fn = torch.mean
            elif self.out_method == 'max':
                fn = lambda x, d: torch.max(x, dim=d)[0]
            else:
                raise NotImplementedError
            out_embed = fn(torch.stack(list_graph_embed), 0)
            out_node_embed = fn(torch.stack(list_node_embed), 0)

        return out_embed, (g_idx, out_node_embed)

