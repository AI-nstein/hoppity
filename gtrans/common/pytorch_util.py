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
from tqdm import tqdm

from gtrans.common.consts import NONLINEARITIES

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def orthogonal_gru(t):
    assert len(t.size()) == 2
    assert t.size()[0] == 3 * t.size()[1]
    hidden_dim = t.size()[1]

    x0 = torch.Tensor(hidden_dim, hidden_dim)
    x1 = torch.Tensor(hidden_dim, hidden_dim)
    x2 = torch.Tensor(hidden_dim, hidden_dim)

    nn.init.orthogonal_(x0)
    nn.init.orthogonal_(x1)
    nn.init.orthogonal_(x2)

    t[0:hidden_dim, :] = x0
    t[hidden_dim:2*hidden_dim, :] = x1
    t[2*hidden_dim:3*hidden_dim, :] = x2

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
        print('a Parameter inited')
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)
        print('a Linear inited')
    elif isinstance(m, nn.GRU):
        for k in range(m.num_layers):
            getattr(m,'bias_ih_l%d'%k).data.zero_()
            getattr(m,'bias_hh_l%d'%k).data.zero_()
            glorot_uniform(getattr(m,'weight_ih_l%d'%k).data)
            orthogonal_gru(getattr(m,'weight_hh_l%d'%k).data)
        print('a GRU inited')
    elif isinstance(m, nn.GRUCell):
        getattr(m,'bias_ih').data.zero_()
        getattr(m,'bias_hh').data.zero_()
        glorot_uniform(getattr(m,'weight_ih').data)
        orthogonal_gru(getattr(m,'weight_hh').data)        
        print('a GRUCell inited')

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)

class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
        super(MLP, self).__init__()
        self.act_last = act_last
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.bn = bn

        if isinstance(hidden_dims, str):
            hidden_dims = list(map(int, hidden_dims.split("-")))
        assert len(hidden_dims)
        hidden_dims = [input_dim] + hidden_dims
        self.output_size = hidden_dims[-1]
        
        list_layers = []

        for i in range(1, len(hidden_dims)):
            list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i + 1 < len(hidden_dims):  # not the last layer
                if self.bn:
                    bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
                    list_layers.append(bnorm_layer)
                list_layers.append(NONLINEARITIES[self.nonlinearity])
                if dropout > 0:
                    list_layers.append(nn.Dropout(dropout))
            else:
                if act_last is not None:
                    list_layers.append(NONLINEARITIES[act_last])

        self.main = nn.Sequential(*list_layers)

    def forward(self, z):
        x = self.main(z)
        return x
