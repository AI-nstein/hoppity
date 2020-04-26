from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
from torchext import replace_rows
import torch.nn as nn

from gtrans.common.code_graph import CodeGraph
from gtrans.graphnet.graph_embed import get_gnn
from gtrans.common.dataset import Dataset, get_gnn_graph
from gtrans.common.pytorch_util import MLP
from gtrans.model.graph_ops import get_graph_op, GraphOp
from gtrans.common.consts import OP_ADD_NODE, OP_NONE, OP_DEL_NODE, OP_REPLACE_TYPE, OP_REPLACE_VAL, CONTENT_NODE_TYPE, t_float, DEVICE


class GraphTrans(GraphOp):
    def __init__(self, args):
        super(GraphTrans, self).__init__(args)
        self.max_modify_steps = args.max_modify_steps
        self.gnn_type = args.gnn_type

        self.gnn = get_gnn(args, node_dim=Dataset.num_node_types())

        self.op_names = [OP_NONE, OP_REPLACE_VAL, OP_DEL_NODE, OP_REPLACE_TYPE, OP_ADD_NODE]
        ops = [get_graph_op(name, args) for name in self.op_names]
        self.ops = nn.ModuleList(ops)
        self.op_pred = MLP(input_dim=args.latent_dim, hidden_dims=[args.latent_dim, len(ops)],
                           nonlinearity=args.act_func)
        self.op_embedding = nn.Embedding(len(ops), args.latent_dim)
        self.cached_masks = {}

    def get_cached_mask(self, i):
        if not i in self.cached_masks:
            self.cached_masks[i] = torch.zeros(len(self.op_names), dtype=t_float).to(DEVICE)

        return self.cached_masks[i]

    def build_masks(self, possible_ops):
        masks = []
        for i, ops in enumerate(possible_ops):
            idxs = [self.op_names.index(op) for op in ops]
            m = self.get_cached_mask(i)
            m.zero_()
            m[idxs] = 1.0

            masks.append(m)

        return torch.stack(masks)

    def _is_root(self, node, ast):
        if node.index >= ast.num_nodes:
            return False
        parent = ast.nodes[node.index].parent
        return parent is None

    def get_possible_ops(self, node, ast):
        possible_ops = [OP_ADD_NODE, OP_DEL_NODE, OP_REPLACE_TYPE, OP_REPLACE_VAL, OP_NONE]

        if self._is_root(node, ast):
            possible_ops.remove(OP_DEL_NODE)

        if node.name is None and not node.ast_node.is_leaf:
            possible_ops.remove(OP_REPLACE_VAL)

        return possible_ops

    def set_rand_flag(self, val):
        self.rand_flag = val
        for op in self.ops:
            if op is not None:
                op.rand_flag = self.rand_flag

    def choose_op(self, targets=None, masks=None, beam_size=1):
        target_indices = None
        if targets is not None:
            target_indices = [self.op_names.index(t) for t in targets]
        logits = self.op_pred(self.states)

        num_tries = min(beam_size, len(self.op_names))
        self._get_fixdim_choices(logits,
                                 self.op_embedding,
                                 target_indices=target_indices,
                                 masks=masks,
                                 beam_size=beam_size,
                                 num_tries=num_tries)
        if target_indices is None:
            target_indices = [t[-1] for t in self.hist_choices]
        return np.array(target_indices)

    def update_graph_repre(self, gnn_graphs):
        graph_embed, (_, node_embedding) = self.gnn(gnn_graphs)
        self.states = self.cell(graph_embed, self.states)
        return node_embedding

    def select_samples(self, graph_list, states, node_embedding, sample_ids):
        if len(sample_ids) == len(graph_list):
            return self.ll, states, node_embedding, self.sample_indices

        part_states = states[sample_ids]

        prefix_sum = []
        offset = 0
        for i, graph in enumerate(graph_list):
            prefix_sum.append(offset)
            offset += graph.pg.num_nodes
        assert offset == node_embedding.shape[0]

        node_idx = []
        for i in sample_ids:
            offset = prefix_sum[i]
            node_idx += list(range(offset, offset + graph_list[i].pg.num_nodes))
        selected = set(sample_ids)
        sub_ids = []
        sub_num = 0
        for l in self.sample_indices:
            sub_l = [idx for idx in l if idx in selected]
            if len(sub_l):
                sub_ids.append(list(range(sub_num, sub_num + len(sub_l))))
                sub_num += len(sub_l)
        assert sub_num == len(sample_ids)

        return self.ll[sample_ids], part_states, node_embedding[node_idx], sub_ids

    def merge_ops(self, sub_infos, beam_size):
        list_cands = [[] for _ in range(len(self.sample_indices))]
        batch_map = {}
        for i, l in enumerate(self.sample_indices):
            for j in l:
                batch_map[j] = i
        for sub_idx, (sub_new_asts, sub_new_refs, sub_ll, sub_states, sub_ids, sample_ids) in enumerate(sub_infos):
            prev_batch = -1
            bid_list = []
            for i, sid in enumerate(sample_ids):
                bid = batch_map[sid]
                if sub_ids is None:
                    cand_info = (sub_ll[i], sub_idx, i)
                    list_cands[bid].append(cand_info)
                elif bid != prev_batch:
                    prev_batch = bid
                    bid_list.append(bid)
            if sub_ids is None:
                continue
            assert len(bid_list) == len(sub_ids)
            for i in range(len((sub_ids))):
                bid = bid_list[i]
                for j in sub_ids[i]:
                    cand_info = (sub_ll[j], sub_idx, j)
                    list_cands[bid].append(cand_info)
        ll_list = []
        new_asts = []
        new_refs = []
        list_states = []
        sample_indices = []
        contents = []
        total_new = 0
        for bid, cands in enumerate(list_cands):
            cands.sort(key=lambda x: -x[0])
            cands = cands[:beam_size]
            for new_ll, sub_idx, j in cands:
                ll_list.append(new_ll)
                sub_new_asts, sub_new_refs, _, sub_states, _, _ = sub_infos[sub_idx]
                new_asts.append(sub_new_asts[j])
                new_refs.append(sub_new_refs[j])
                list_states.append(sub_states[j])
                contents.append(None)
            sample_indices.append(list(range(total_new, total_new + len(cands))))
            total_new += len(cands)
        self.ll = np.array(ll_list, dtype=np.float32)
        self.states = torch.stack(list_states)
        self.sample_indices = sample_indices
        self.node_embedding = None  # the existing state is now obsolete
        self.sample_buf = None  # the existing sample buf is now obsolete
        return new_asts, new_refs, contents

    def forward(self, sample_list, phase, beam_size=1, pred_gt=False, op_given=False, loc_given=False):
        """
        Args:
            sample_list: list of samples
            phase: 'train', 'val', 'test', etc
            beam_size: size of beam search
            pred_gt: output groundtruth, instead of making prediction
                    when training, this must be true
        """
        self.sample_indices = [[i] for i in range(len(sample_list))]
        if pred_gt:
            assert beam_size == 1
        if phase == 'train':
            assert pred_gt
            self.train()
        else:
            self.eval()
        self.states = None
        self.ll = 0
        batch_size = len(sample_list)
        self.sample_buf = [s.buggy_gnn_graph for s in sample_list]


        for step in range(self.max_modify_steps):
            self.hist_choices = None
            self.node_embedding = self.update_graph_repre(self.sample_buf)

            self.cur_edits = [None] * len(self.sample_buf)
            if not phase == "test" or pred_gt or loc_given:
                for i in range(batch_size):
                    s = sample_list[i]
                    if step < len(s.g_edits):
                        for j in self.sample_indices[i]:
                            self.cur_edits[j] = s.g_edits[step]

            # predict location
            target_indices = None
            if pred_gt or loc_given:
                target_indices = []
                for i, g in enumerate(self.sample_buf):

                    e = self.cur_edits[i]

                    if not e:
                        target_indices.append(g.pg.ast.root_node.index)
                        continue

                    if e.op == OP_ADD_NODE:
                        if not hasattr(e, 'parent_id'):
                            target_indices = None
                            break

                        target_indices.append(e.parent_id)

                    elif e.op == OP_NONE:
                        target_indices.append(g.pg.ast.root_node.index)
                    else:
                        if not hasattr(e, 'node_index'):
                            target_indices = None
                            break
                        target_indices.append(e.node_index)


            self.get_node(self.node_embedding,
                          fn_node_select=lambda bid, node: node.node_type != CONTENT_NODE_TYPE,
                          target_indices=target_indices,
                          beam_size=beam_size)

            if pred_gt or op_given:
                target_op = [t.op if t is not None else OP_NONE for t in self.cur_edits]
            else:
                target_op = None

            target_nodes = [choice[0] for choice in self.hist_choices]

            target_asts = [s.pg.ast for s in self.sample_buf]

            possible_ops = [self.get_possible_ops(node, ast) for node, ast in zip(target_nodes, target_asts)]

            masks = self.build_masks(possible_ops)

            op_indices = self.choose_op(target_op, masks=masks, beam_size=beam_size)

            new_asts = [s.pg.ast for s in self.sample_buf]
            new_refs = [s.pg.refs for s in self.sample_buf]

            num_stops = len(np.where(op_indices == self.op_names.index(OP_NONE))[0])
            if num_stops == len(self.sample_buf):  # no one needs further edits
                break

            sub_infos = []
            for op_id in range(len(self.ops)):

                sample_ids = np.where(op_indices == op_id)[0]
                if len(sample_ids) == 0:
                    continue
                if self.op_names[op_id] != OP_NONE:
                    sub_ll, sub_states, sub_node_embedding, sub_ids = self.select_samples(graph_list=self.sample_buf,
                                                                                          states=self.states,
                                                                                          node_embedding=self.node_embedding,
                                                                                          sample_ids=sample_ids)


                    sub_gnn_graphs = [self.sample_buf[i] for i in sample_ids]
                    sub_edits = [self.cur_edits[i] for i in sample_ids]
                    sub_hists = [self.hist_choices[i] for i in sample_ids]

                    sub_new_asts, sub_ll, sub_states, sub_ids, sub_new_refs = self.ops[op_id](sub_ll,
                                                                                sub_states,
                                                                                sub_gnn_graphs,
                                                                                sub_node_embedding,
                                                                                sub_ids,
                                                                                sub_edits,
                                                                                sub_hists,
                                                                                beam_size=beam_size,
                                                                                pred_gt=pred_gt,
                                                                                loc_given=loc_given)

                    if len(sample_ids) == len(self.sample_buf): # no need to do gather
                        self.ll = sub_ll
                        self.states = sub_states
                        self.sample_indices = sub_ids
                        new_asts = sub_new_asts
                        new_refs = sub_new_refs
                        break
                    else:
                        if beam_size == 1:
                            for i, idx in enumerate(sample_ids):
                                new_asts[idx] = sub_new_asts[i]
                                new_refs[idx] = sub_new_refs[i]
                            self.ll = replace_rows(self.ll, sample_ids, sub_ll)
                            self.states = replace_rows(self.states, sample_ids, sub_states)
                        else:
                            sub_infos.append((sub_new_asts, sub_new_refs, sub_ll, sub_states, sub_ids, sample_ids))
                elif beam_size > 1:
                    sub_new_asts = [new_asts[i] for i in sample_ids]
                    sub_new_refs = [new_refs[i] for i in sample_ids]
                    sub_ll = self.ll[sample_ids]
                    sub_states = self.states[sample_ids]
                    sub_ids = None
                    sub_infos.append((sub_new_asts, sub_new_refs, sub_ll, sub_states, sub_ids, sample_ids))
            if beam_size > 1 and len(sub_infos):
                new_asts, new_refs, contents = self.merge_ops(sub_infos, beam_size)
            else:
                contents = [s.pg.contents for s in self.sample_buf]
            if step + 1 == self.max_modify_steps:
                break

            # build new graph representation, after the current edit to the graph
            buggy_code_graphs = [CodeGraph(ast, new_ref, c) for (ast, new_ref, c) in zip(new_asts, new_refs, contents)]
            self.sample_buf = [get_gnn_graph(cg, self.gnn_type) for cg in buggy_code_graphs]

        agg_asts = []
        tot_asts = 0
        for l in self.sample_indices:
            cur_asts = [new_asts[i] for i in l]
            agg_asts.append(cur_asts)
            tot_asts += len(cur_asts)
        assert tot_asts == len(new_asts)

        return self.ll, agg_asts
