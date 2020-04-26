from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from copy import deepcopy
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from gtrans.common.code_graph import AstNode
from gtrans.common.consts import OP_ADD_NODE, OP_NONE, CONTENT_NODE_TYPE, OP_DEL_NODE, OP_REPLACE_TYPE, OP_REPLACE_VAL, SEPARATOR
from gtrans.common.consts import DEVICE, t_float
from gtrans.common.dataset import Dataset, GraphEditCmd
from gtrans.common.pytorch_util import MLP, glorot_uniform
from gtrans.common.code_graph import CgNode
from gtrans.model.utils import AutoregModel, adjust_refs, adjust_attr_order, get_randk, get_rand_one
from torchext import jagged_log_softmax, jagged_argmax, jagged_topk, jagged_append, jagged2padded

class GraphOp(AutoregModel):
    def __init__(self, args, op_name=None):
        super(GraphOp, self).__init__()
        self.op_name = op_name
        self.ll, self.states = None, None
        self.hist_choices, self.sample_buf, self.cur_edits, self.sample_indices, self.node_embedding = None, None, None, None, None

        if args.rnn_cell == 'gru':
            self.cell = nn.GRUCell(args.latent_dim, args.latent_dim)
        else:
            raise NotImplementedError

        if args.comp_method == "inner_prod":
            self.comp_func = lambda  x, y: torch.sum(x * y, dim=1).view(-1)
        elif args.comp_method == "mlp":
            self.pred = MLP(2 * args.latent_dim, [2 * args.latent_dim] * 2 + [1])
            self.comp_func = lambda x, y: self.pred(torch.cat((x, y), dim=1)).view(-1)
        elif args.comp_method == "bilinear":
            self.bilin = nn.Bilinear(args.latent_dim, args.latent_dim, 1)
            self.comp_func = lambda x, y: self.bilin(x, y).view(-1)
        elif args.comp_method == "multilin":
            self.bilin = nn.Bilinear(args.latent_dim, args.latent_dim, 16)
            self.pred = MLP(16, [args.latent_dim, 1])
            self.comp_func = lambda x, y: self.pred(self.bilin(x, y)).view(-1)
        else:
            raise NotImplementedError

    def _get_num_options(self, th_prefix_sum):
        tmp_prefix = th_prefix_sum.cpu().data.numpy()
        num_options = []
        for i in range(tmp_prefix.shape[0]):
            if i == 0:
                num_options.append(tmp_prefix[0])
            else:
                num_options.append(tmp_prefix[i] - tmp_prefix[i - 1])
        return num_options

    def get_node(self, node_embedding, fn_node_select,
                 target_indices=None,
                 const_node_embeds=None,
                 const_nodes=None,
                 fn_const_node_pred=None,
                 beam_size=1):
        if target_indices is not None:
            beam_size = 1
        node_indices = []
        offset = 0
        local_target_indices = None if target_indices is None else []
        num_const_nodes = 0 if const_node_embeds is None else const_node_embeds.shape[0]

        node_of_graph_idx = []
        list_prefix_sum = []
        old_offsets = []
        for g_idx, gnn_g in enumerate(self.sample_buf):
            cur_size = len(node_indices)
            old_offsets.append(offset)
            for i, node in enumerate(gnn_g.pg.node_list):
                if fn_node_select(g_idx, node):
                    if local_target_indices is not None and i == target_indices[g_idx]:
                        local_target_indices.append(len(node_indices) + g_idx * num_const_nodes)
                    node_indices.append(offset + i)
            if local_target_indices is not None and target_indices[g_idx] and target_indices[g_idx] >= gnn_g.num_nodes:  # this is referring to a global const node
                local_target_indices.append(len(node_indices) + g_idx * num_const_nodes + target_indices[g_idx] - gnn_g.num_nodes)
            num_candidates = len(node_indices) - cur_size

            node_of_graph_idx += [g_idx] * num_candidates
            list_prefix_sum.append(len(node_indices))
            offset += gnn_g.pg.num_nodes
        prefix_sum = torch.LongTensor(list_prefix_sum).to(node_embedding.device)
        node_of_graph_idx = torch.LongTensor(node_of_graph_idx).to(node_embedding.device)
        node_embedding = torch.index_select(node_embedding, 0, torch.LongTensor(node_indices).to(node_embedding.device))

        # do the inner product
        rep_states = torch.index_select(self.states, 0, node_of_graph_idx)
        if rep_states.shape[0]:
            logits = self.comp_func(rep_states, node_embedding)
        else:
            logits = None
        if num_const_nodes:
            consts_logits = fn_const_node_pred(self.states)
            if logits is None:
                logits = consts_logits.view(-1)
            else:
                logits = jagged_append(logits, prefix_sum, consts_logits)
            prefix_correction = [num_const_nodes * (i + 1) for i in range(len(self.sample_buf))]
            prefix_correction = torch.LongTensor(prefix_correction).to(node_embedding.device)
            prefix_sum = prefix_sum + prefix_correction

        ll_all = jagged_log_softmax(logits, prefix_sum)
        if beam_size == 1:
            if local_target_indices is None:
                if self.rand_flag:
                    num_options = self._get_num_options(prefix_sum)
                    indices = [np.random.randint(i) for i in num_options]
                    indices = torch.LongTensor(indices).to(ll_all.device)
                else:
                    indices = jagged_argmax(ll_all, prefix_sum)
                indices[1:] += prefix_sum[:-1]
                indices = indices.detach()
            else:
                assert len(local_target_indices) == len(self.sample_buf)
                indices = torch.LongTensor(local_target_indices).to(node_embedding.device)
            ll_to_add = ll_all[indices]
            if isinstance(self.ll, np.ndarray):
                ll_to_add = ll_to_add.cpu().data.numpy()
            self.ll = self.ll + ll_to_add
            prefix_sum = [0] + prefix_sum.data.cpu().numpy().tolist()[:-1]
            old_sample_ids = range(len(self.sample_buf))
        else:
            if self.rand_flag:
                pad_val = np.finfo(np.float32).min
                padded_val = jagged2padded(ll_all, prefix_sum, pad_val)
                num_options = self._get_num_options(prefix_sum)
                op_lim = min(num_options)
                k = min(op_lim, beam_size)
                indices = []
                step_ll = []
                np_ll = padded_val.cpu().data.numpy()
                for i in enumerate(num_options):
                    x_p = list(range(num_options[i]))
                    random.shuffle(x_p)
                    li = x_p[:k]
                    ls = []
                    for j in li:
                        ls.append(np_ll[i, j])
                    step_ll.append(ls)
                    indices.append(li)
                step_ll = torch.FloatTensor(step_ll).to(ll_all.device)
                step_choices = torch.LongTensor(indices).to(ll_all.device)
            else:
                step_ll, step_choices = jagged_topk(ll_all, prefix_sum, beam_size)

            indices, old_sample_ids = self.update_beam_stats(step_choices, step_ll, beam_size, prefix_sum)
            buf_prefix = [0] + list_prefix_sum
            prefix_sum = [buf_prefix[i] + i * num_const_nodes for i in old_sample_ids]
            indices += torch.LongTensor(prefix_sum).to(node_embedding.device)

        indices = indices.cpu().data.numpy()

        if num_const_nodes:
            new_indices = []
            for g_idx, gnn_g in enumerate(self.sample_buf):
                cur_local_idx = indices[g_idx] - prefix_sum[g_idx]
                sample_id = old_sample_ids[g_idx]
                num_candidates = list_prefix_sum[sample_id] - list_prefix_sum[sample_id - 1] if sample_id else list_prefix_sum[sample_id]
                if cur_local_idx >= num_candidates:  # this is a choice for global const node
                    new_indices.append(cur_local_idx - num_candidates + node_embedding.shape[0])
                else:  # this is a local choice
                    new_indices.append(list_prefix_sum[sample_id] - num_candidates + cur_local_idx)
            indices = new_indices
            update_embedding = torch.cat((node_embedding, const_node_embeds), dim=0)
        else:
            update_embedding = node_embedding
        self.states = self.cell(update_embedding[indices], self.states)

        nodes = []
        list_choices = []
        for g_idx, gnn_g in enumerate(self.sample_buf):
            if indices[g_idx] >= node_embedding.shape[0]:  # this is a global const choice
                node_idx = indices[g_idx] - node_embedding.shape[0]
                new_node = const_nodes[node_idx]
            else:
                node_idx = node_indices[indices[g_idx]] - old_offsets[old_sample_ids[g_idx]]
                new_node = gnn_g.pg.node_list[node_idx]
            nodes.append(new_node)
            hist_nodes = [] if self.hist_choices is None else self.hist_choices[old_sample_ids[g_idx]]
            list_choices.append(hist_nodes + [new_node])
        self.hist_choices = list_choices
        return nodes

    def _get_fixdim_choices(self, logits, update_embeddings, target_indices=None, masks=None, beam_size=1, num_tries=1):
        if target_indices is not None:
            beam_size = 1
        if masks is not None:
            logits = logits * masks + (1.0 - masks) * -10000000
        ll_all = F.log_softmax(logits, dim=1)
        if beam_size == 1:
            if target_indices is None:
                if self.rand_flag:
                    indices = get_rand_one(ll_all)
                else:
                    indices = torch.argmax(ll_all, dim=-1)
            else:
                indices = torch.LongTensor(target_indices).to(logits.device)
            ll_to_add = ll_all.gather(1, indices.view(-1, 1)).view(-1)
            if isinstance(self.ll, np.ndarray):
                ll_to_add = ll_to_add.cpu().data.numpy()
            self.ll = self.ll + ll_to_add
            old_sample_ids = range(len(self.sample_buf))
        else:
            if self.rand_flag:
                step_ll, indices = get_randk(ll_all, num_tries)
            else:
                step_ll, indices = torch.topk(ll_all, num_tries, dim=-1)
            indices, old_sample_ids = self.update_beam_stats(indices, step_ll, beam_size)

        updates = update_embeddings(indices)
        self.states = self.cell(updates, self.states)

        if target_indices is None:
            target_indices = indices.cpu().data.numpy()
        list_choices = []
        for g_idx in range(len(self.sample_buf)):
            hist_opts = [] if self.hist_choices is None else self.hist_choices[old_sample_ids[g_idx]]
            list_choices.append(hist_opts + [target_indices[g_idx]])
        self.hist_choices = list_choices
        return target_indices

    def setup_forward(self, ll, states, gnn_graphs, node_embedding, sample_indices, hist_choices, init_edits):
        self.ll = ll
        self.states = states
        self.sample_buf = gnn_graphs
        self.sample_indices = sample_indices
        self.node_embedding = node_embedding
        self.hist_choices = hist_choices
        self.cur_edits = init_edits


class TypedGraphOp(GraphOp):
    def __init__(self, args, op_name):
        super(TypedGraphOp, self).__init__(args, op_name=op_name)
        self.node_type_pred = MLP(input_dim=args.latent_dim,
                                  hidden_dims=[args.latent_dim, Dataset.num_node_types()],
                                  nonlinearity=args.act_func)
        self.node_type_embedding = nn.Embedding(Dataset.num_node_types(), args.latent_dim)

    def get_node_type(self, target_types=None, beam_size=1):
        target_indices = None
        if target_types is not None:
            target_indices = [Dataset.get_id_of_ntype(t) for t in target_types]
        logits = self.node_type_pred(self.states)
        num_tries = min(beam_size, Dataset.num_node_types())
        self._get_fixdim_choices(logits,
                                 self.node_type_embedding,
                                 target_indices=target_indices,
                                 masks=None, 
                                 beam_size=beam_size, 
                                 num_tries=num_tries)
        found_types = []
        for i in range(len(self.hist_choices)):
            t = Dataset.get_ntype_of_id(self.hist_choices[i][-1])
            self.hist_choices[i][-1] = t
            found_types.append(t)
        return found_types if target_types is None else target_types

class AddNodeOp(TypedGraphOp):
    def __init__(self, args):
        super(AddNodeOp, self).__init__(args, op_name=OP_ADD_NODE)
        v = torch.zeros(Dataset.num_const_values(), args.latent_dim, dtype=t_float)
        glorot_uniform(v)
        self.const_name_embedding = Parameter(v)

        self.const_name_pred = MLP(input_dim=args.latent_dim, 
                                   hidden_dims=[args.latent_dim, Dataset.num_const_values()],
                                   nonlinearity=args.act_func)

    def forward(self, ll, states, gnn_graphs, node_embedding, sample_indices, init_edits, hist_choices, beam_size=1, pred_gt=False, loc_given=False):
        self.setup_forward(ll, states, gnn_graphs, node_embedding, sample_indices, hist_choices, init_edits)

        # self.node_type, self.node_name, self.parent_id, self.child_rank
        target_types = [e.node_type for e in self.cur_edits] if pred_gt else None
        self.get_node_type(target_types, beam_size=beam_size)

        # select node name
        if pred_gt:
            target_name_indices = []
            for i, g in enumerate(self.sample_buf):
                if self.cur_edits[i].node_name in g.pg.contents:
                    target_name_idx = g.pg.contents[self.cur_edits[i].node_name].index
                elif self.cur_edits[i].node_name in Dataset.get_value_vocab():
                    target_name_idx = g.num_nodes + Dataset.get_value_vocab()[self.cur_edits[i].node_name]
                else:
                    raise NotImplementedError
                target_name_indices.append(target_name_idx)
        else:
            target_name_indices = None

        self.get_node(self.node_embedding,
                      fn_node_select=lambda bid, node: node.node_type == CONTENT_NODE_TYPE,
                      target_indices=target_name_indices,
                      const_node_embeds=self.const_name_embedding,
                      const_nodes=Dataset._id_value_map,
                      fn_const_node_pred=self.const_name_pred,
                      beam_size=beam_size)

        parent_nodes = [self.hist_choices[i][0] for i in range(len(self.hist_choices))]

        for e in self.cur_edits:
            if not hasattr(e, 'child_rank'):
                loc_given = False
                break

        # select left sibling
        if pred_gt or loc_given:
            target_prenode_ids = []
            for i, g in enumerate(self.sample_buf):
                if self.cur_edits[i].child_rank == 0:
                    target_prenode_ids.append(parent_nodes[i].index)
                else:
                    ch_idx = self.cur_edits[i].child_rank - 1
                    par_node = parent_nodes[i].ast_node
                    target_prenode_ids.append(par_node.children[ch_idx].index)
        else:
            target_prenode_ids = None

        self.get_node(self.node_embedding,
                      fn_node_select=lambda bid, node: self._sibling_filter(node, self.sample_buf[bid].pg.ast, parent_nodes[bid].index),
                      target_indices=target_prenode_ids,
                      beam_size=beam_size)

        new_asts = []
        new_refs = []
        for i, g in enumerate(self.sample_buf):
            ast = deepcopy(g.pg.ast)

            parent_node, _, node_type, target_name, pre_node = self.hist_choices[i]
            if isinstance(target_name, CgNode):
                target_name = target_name.name
            if pre_node.index == parent_node.index:
                ch_rank = 0
            else:
                ch_rank = parent_node.ast_node.child_rank(pre_node.ast_node) + 1

            new_node = AstNode(node_type=node_type, value=target_name)
            ast.add_node(new_node)

            p_node = ast.nodes[parent_node.index]
            p_node.add_child(new_node, ch_rank)
            new_asts.append(ast)
            new_refs.append(g.pg.refs)

            if not target_name:
                target_name = "None"

            if SEPARATOR in node_type:
                tmp_type = node_type.split(SEPARATOR)[0]
            else:
                tmp_type = node_type
            e = GraphEditCmd(SEPARATOR.join([OP_ADD_NODE, str(parent_node.index), str(ch_rank), tmp_type, target_name]))
            e.node_type = node_type
            ast.append_edit(e)

        return new_asts, self.ll, self.states, self.sample_indices, new_refs

    def _sibling_filter(self, node, ast, parent_idx):
        if node.index >= ast.num_nodes:
            return False
        if node.index == parent_idx:
            return True
        parent = ast.nodes[node.index].parent
        if parent is None:
            return False
        return parent.index == parent_idx


class DelNodeOp(GraphOp):
    def __init__(self, args):
        super(DelNodeOp, self).__init__(args, op_name=OP_DEL_NODE)

    def forward(self, ll, states, gnn_graphs, node_embedding, sample_indices, init_edits, hist_choices, beam_size=1, pred_gt=False, loc_given=False):
        self.setup_forward(ll, states, gnn_graphs, node_embedding, sample_indices, hist_choices, init_edits)

        new_asts = []
        new_refs = []
        for i, g in enumerate(self.sample_buf):
            ast = deepcopy(g.pg.ast)
            target_node = self.hist_choices[i][0]

            ast.remove_node(ast.nodes[target_node.index])
            g.pg.refs = adjust_refs(deepcopy(g.pg.refs), target_node.index)
            new_asts.append(ast)
            new_refs.append(g.pg.refs)
            ast.append_edit(GraphEditCmd(SEPARATOR.join([OP_DEL_NODE, str(target_node.index)])))

        return new_asts, self.ll, self.states, self.sample_indices, new_refs

    def _is_root(self, node, ast):
        if node.index >= ast.num_nodes:
            return False
        parent = ast.nodes[node.index].parent
        return parent is None


class ReplaceTypeOp(TypedGraphOp):
    def __init__(self, args):
        super(ReplaceTypeOp, self).__init__(args, op_name=OP_REPLACE_TYPE)
        self.cached_masks = {}

    def get_cached_mask(self, i):
        if not i in self.cached_masks:
            self.cached_masks[i] = torch.zeros(Dataset.num_node_types(), dtype=t_float).to(DEVICE)
        return self.cached_masks[i]

    def _build_masks(self, list_avails):
        masks = []
        for i, avails in enumerate(list_avails):
            m = self.get_cached_mask(i)
            m.zero_()
            if avails is None or len(avails) == 0:
                m = m + 1.0
            else:
                m[avails] = 1.0
            masks.append(m)
        return torch.stack(masks)

    def forward(self, ll, states, gnn_graphs, node_embedding, sample_indices, init_edits, hist_choices, beam_size=1, pred_gt=False, loc_given=False):
        self.setup_forward(ll, states, gnn_graphs, node_embedding, sample_indices, hist_choices, init_edits)

        target_node_types = [e.node_type for e in self.cur_edits] if pred_gt else None

        self.get_node_type(target_node_types, beam_size=beam_size)

        new_asts = []
        new_refs = []
        for i, g in enumerate(self.sample_buf):
            ast = deepcopy(g.pg.ast)
            target_node, _, target_type = self.hist_choices[i]
            node_to_be_edit = ast.nodes[target_node.index]

            if SEPARATOR in target_node.node_type:
                target_type = SEPARATOR.join([target_node.node_type.split(SEPARATOR)[0], target_type])

            node_to_be_edit.node_type = target_type
            node_to_be_edit.children = adjust_attr_order(target_type, node_to_be_edit.children)
            new_asts.append(ast)
            new_refs.append(g.pg.refs)

            ast.append_edit(GraphEditCmd(SEPARATOR.join([OP_REPLACE_TYPE, str(target_node.index), target_type])))

        return new_asts, self.ll, self.states, self.sample_indices, new_refs


class ReplaceValOp(GraphOp):
    def __init__(self, args):
        super(ReplaceValOp, self).__init__(args, op_name=OP_REPLACE_VAL)
        v = torch.zeros(Dataset.num_const_values(), args.latent_dim, dtype=t_float)
        glorot_uniform(v)
        self.const_name_embedding = Parameter(v)

        self.const_name_pred = MLP(input_dim=args.latent_dim, 
                                   hidden_dims=[args.latent_dim, Dataset.num_const_values()],
                                   nonlinearity=args.act_func)


    def forward(self, ll, states, gnn_graphs, node_embedding, sample_indices, init_edits, hist_choices, beam_size=1, pred_gt=False, loc_given=False):
        self.setup_forward(ll, states, gnn_graphs, node_embedding, sample_indices, hist_choices, init_edits)

        # select node name
        if pred_gt:
            target_name_indices = []
            for i, g in enumerate(self.sample_buf):
                if self.cur_edits[i].node_name in g.pg.contents:
                    target_name_idx = g.pg.contents[self.cur_edits[i].node_name].index
                elif self.cur_edits[i].node_name in Dataset.get_value_vocab():
                    target_name_idx = g.num_nodes + Dataset.get_value_vocab()[self.cur_edits[i].node_name]
                else:
                    raise NotImplementedError
                target_name_indices.append(target_name_idx)
        else:
            target_name_indices = None
        self.get_node(self.node_embedding,
                      fn_node_select=lambda bid, node: node.node_type == CONTENT_NODE_TYPE,
                      target_indices=target_name_indices,
                      const_node_embeds=self.const_name_embedding,
                      const_nodes=Dataset._id_value_map,
                      fn_const_node_pred=self.const_name_pred,
                      beam_size=beam_size)

        new_asts = []
        new_refs = []
        for i, g in enumerate(self.sample_buf):
            ast = deepcopy(g.pg.ast)
            target_node, _, target_name = self.hist_choices[i]
            if isinstance(target_name, CgNode):
                target_name = target_name.name
            node_to_be_edit = ast.nodes[target_node.index]
            node_to_be_edit.value = target_name
            g.pg.refs = adjust_refs(deepcopy(g.pg.refs), target_node.index)
            new_asts.append(ast)
            new_refs.append(g.pg.refs)
            target_name = "None" if not target_name else target_name
            ast.append_edit(GraphEditCmd(SEPARATOR.join([OP_REPLACE_VAL, str(target_node.index), target_name])))

        return new_asts, self.ll, self.states, self.sample_indices, new_refs


def get_graph_op(name, args):
    if name == OP_ADD_NODE:
        return AddNodeOp(args)
    elif name == OP_DEL_NODE:
        return DelNodeOp(args)
    elif name == OP_REPLACE_TYPE:
        return ReplaceTypeOp(args)
    elif name == OP_REPLACE_VAL:
        return ReplaceValOp(args)
    elif name == OP_NONE:
        return None
    else:
        raise NotImplementedError
