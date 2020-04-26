from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import sys
import random
import pickle as cp
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from gtrans.data_process import build_ast
from gtrans.common.consts import SEPARATOR, ADDITIONAL_NODES, shift_node_types
from gtrans.common.consts import OP_ADD_NODE, OP_DEL_NODE, OP_REPLACE_TYPE, OP_REPLACE_VAL, OP_NONE
from gtrans.common.code_graph import CodeGraph
from gtrans.graphnet.s2v_lib import Code2InvGraph, MergedGraph, MultiGraph
from gtrans.data_process.utils import code_group_generator, get_bug_prefix, clean_unknown


class GraphEditCmd(object):
    def __init__(self, cmd_txt):
        cmds = cmd_txt.split(SEPARATOR)
        self.op = cmds[0]
        if self.op == OP_REPLACE_TYPE:
            #assert len(cmds) == 3
            self.node_index = cmds[1]
            self.node_index = int(self.node_index)

            self.node_type = cmds[2:]
            if isinstance(self.node_type, list):
                self.node_type = SEPARATOR.join(self.node_type)

        elif self.op == OP_ADD_NODE:
            assert len(cmds) >= 5
            self.parent_id, self.child_rank = cmds[1:3]

            if len(cmds) > 5:
                self.node_type = SEPARATOR.join([cmds[3], cmds[4]])
                idx = 5
            else:
                self.node_type = cmds[3]
                idx = 4

            self.clean_name = cmds[idx]
            self.node_name = self.clean_name
            self.parent_id = int(self.parent_id)
            self.child_rank = int(self.child_rank)
            self.children = []

            assert len(cmds) <= 6  # no support for multiple children
            '''
            if len(cmds) > 5:
                for i in range(5, len(cmds), 2):
                    obj = {}
                    obj["node_type"] = cmds[i]
                    obj["node_value"] = cmds[i+1]
                    if obj["node_value"] == "None":
                        obj["node_value"] = None

                    self.children.append(obj)
            '''

            #self.node_index = self.parent_id
        elif self.op == OP_REPLACE_VAL:
            self.node_index, self.node_name = cmds[1:]
            self.node_index = int(self.node_index)
            self.clean_name = self.node_name
        elif self.op == OP_DEL_NODE:
            assert len(cmds) == 2
            self.node_index = int(cmds[1])
        elif self.op == OP_NONE:
            assert len(cmds) == 1
            self.node_index = None
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if self.op != other.op:
            return False
        if self.op == OP_REPLACE_TYPE:
            return self.node_index == other.node_index and self.node_type == other.node_type
        elif self.op == OP_ADD_NODE:
            if self.parent_id != other.parent_id:
                return False
            if self.child_rank != other.child_rank:
                return False
            if self.node_type != other.node_type:
                return False
            if self.clean_name != other.clean_name:
                return False
            return True
        elif self.op == OP_REPLACE_VAL:
            return self.node_index == other.node_index and self.node_name == other.node_name
        elif self.op == OP_DEL_NODE:
            return self.node_index == other.node_index
        elif self.op == OP_NONE:
            return True
        else:
            raise NotImplementedError

    def __str__(self):
        d_edit = {}
        d_edit["op"] = self.op

        if self.op == OP_ADD_NODE:
            node_id = self.parent_id
        else:
            node_id = self.node_index

        d_edit["loc"] = node_id
        d_edit["value"] = None if self.op == OP_REPLACE_TYPE or self.op == OP_DEL_NODE else self.node_name
        d_edit["type"] = None if self.op == OP_REPLACE_VAL or self.op == OP_DEL_NODE else self.node_type
        d_edit["ch_rank"] = self.child_rank if self.op == OP_ADD_NODE else None

        return str(d_edit)

    def clean_unk(self, local_vals):
        if self.op == OP_ADD_NODE or self.op == OP_REPLACE_VAL:
            self.node_name = clean_unknown(local_vals, Dataset.get_value_vocab(), self.clean_name)
            if self.node_name == "None":
                self.node_name = None
    
    def get_fields(self):
        if self.op == OP_DEL_NODE:
            return ["node_index"]
        elif self.op == OP_REPLACE_TYPE:
            return ["node_index", "node_type"]
        elif self.op == OP_REPLACE_VAL:
            return ["node_index", "node_name"]
        elif self.op == OP_ADD_NODE:
            return ["parent_id", "node_type", "node_name"]
        else:
            raise NotImplementedError



def _get_ast_from_file(fname):
    if fname.endswith('pkl'):
        with open(fname, 'rb') as f:
            try:
                ast = cp.load(f)
            except cp.UnpicklingError as e:
                print(e)
                print(fname)
                sys.exit(1)
    else:
        ast = build_ast(fname)
    return ast


class DataSample(object):
    def __init__(self, sample_idx, f_bug, f_fixed, f_diff, b_refs):
        self.index = sample_idx
        self.f_bug = f_bug
        self.f_fixed = f_fixed
        self.f_b_refs = b_refs
        self.f_diff = f_diff
        self._g_edits = None

        self._buggy_ast = None
        self._fixed_ast = None
        self._buggy_refs = None
        self._buggy_code_graph = None
        self._fixed_code_graph = None
        self.buggy_gnn_graph = None
        self.buggy_file = f_bug #save the filename for debugging

    def unload(self):
        self._g_edits = None
        self._buggy_ast = None
        self._fixed_ast = None
        self._buggy_refs = None
        self._fixed_code_graph = None
        self._buggy_code_graph = None
        self.buggy_gnn_graph = None
        
    @property
    def g_edits(self):
        if self._g_edits is None:
            g_edits = []
            with open(self.f_diff, 'r') as f:
                obj = json.load(f)
                for edit in obj:
                    g_edits.append(GraphEditCmd(edit["edit"]))
            self._g_edits = g_edits
        return self._g_edits

    @property
    def buggy_ast(self):
        if self._buggy_ast is None:
            self._buggy_ast = _get_ast_from_file(self.f_bug)
        return self._buggy_ast

    @property
    def fixed_ast(self):
        if self._fixed_ast is None:
            self._fixed_ast = _get_ast_from_file(self.f_fixed)
        return self._fixed_ast

    @property
    def buggy_refs(self):
        if self._buggy_refs is None:
            self._buggy_refs = np.load(self.f_b_refs, allow_pickle=True).item()
        return self._buggy_refs

    @property
    def buggy_code_graph(self):
        if self._buggy_code_graph is None:
            self._buggy_code_graph = CodeGraph(self.buggy_ast, self.buggy_refs)
        return self._buggy_code_graph

    @property
    def fixed_code_graph(self):
        if self._fixed_code_graph is None:
            self._fixed_code_graph = CodeGraph(self.fixed_ast, None)
        return self._fixed_code_graph


def get_gnn_graph(code_graph, gnn_type):
    if gnn_type == 's2v_code2inv':
        return Code2InvGraph(code_graph, Dataset._node_type_dict)
    elif gnn_type == 's2v_multi':
        return MultiGraph(code_graph, Dataset._node_type_dict)
    else:
        return MergedGraph(code_graph, Dataset._node_type_dict)


class Dataset(object):
    _node_type_dict = {}
    _id_ntype_map = {}
    _lang_dict = {}
    _id_lang_map = {}
    _value_vocab = {}
    _id_value_map = {}

    _lang_type = None

    def __init__(self, data_root, gnn_type, data_in_mem=False, resampling=False, sample_types=None, lang_dict=None, valpred_type=None, phases=None):
        self.data_root = data_root
        self.gnn_type = gnn_type
        self.data_in_mem = data_in_mem
        self.resampling = resampling
        self.sample_types = sample_types
        Dataset._lang_type = lang_dict
        print('loading cooked asts and edits')

        self.data_samples = []
        self.sample_index = {}
        self.sample_edit_type = {}
        cooked_gen = code_group_generator(data_root, file_suffix=['_buggy.pkl', '_fixed.pkl', '_gedit.txt', '_refs.npy'])
        if phases is not None:
            avail_set = set()
            for phase in phases:
                idx_file = os.path.join(self.data_root, '%s.txt' % phase)
                if not os.path.isfile(idx_file):
                    continue
                with open(idx_file, 'r') as f:
                    for row in f:
                        sname = row.strip()
                        avail_set.add(sname)

        fidx = 0
        for file_tuple in tqdm(cooked_gen):
            f_bug, f_fixed, f_diff, b_refs = file_tuple
            sample_name = get_bug_prefix(f_bug)
            if  phases is not None and sample_name not in avail_set:
                continue
            if any([not os.path.isfile(f) for f in file_tuple]):
                continue
            
            sample = DataSample(fidx, f_bug, f_fixed, f_diff, b_refs)
            if self.resampling or sample_types is not None:
                s_type = sample.g_edits[0].op
                if sample_types is not None and not s_type in sample_types:
                    continue
                self.sample_edit_type[fidx] = s_type
            self.data_samples.append(sample)
            self.sample_index[sample_name] = fidx
            fidx += 1
        assert len(self.data_samples) == fidx
        print(fidx, 'samples loaded.')

        f_type_vocab = os.path.join(data_root, 'type_vocab.pkl')
        if os.path.isfile(f_type_vocab):
            Dataset.load_type_vocab(f_type_vocab)
        else:
            print('building vocab and saving to', f_type_vocab)
            self.build_node_types()
            with open(f_type_vocab, 'wb') as f:
                d = {}
                d['_node_type_dict'] = Dataset._node_type_dict
                d['_id_ntype_map'] = Dataset._id_ntype_map
                cp.dump(d, f, cp.HIGHEST_PROTOCOL)

        if lang_dict is not None and lang_dict != 'None':
            f_val_dict = os.path.join(data_root, 'val_dict-%s-%s.pkl' % (lang_dict, valpred_type))
            if os.path.isfile(f_val_dict):
                print('loading %s dict from' % lang_dict, f_val_dict)
                with open(f_val_dict, 'rb') as f:
                    d = cp.load(f)
                    Dataset._lang_dict = d['_lang_dict']
                    Dataset._id_lang_map = d['_id_lang_map']
            else:
                print('building %s dict and saving to' % lang_dict, f_val_dict)
                for s in tqdm(self.data_samples):
                    for e in s._gedits:
                        e.clean_unk(s.buggy_code_graph.contents)
                        val = e.node_name if valpred_type == 'node_name' else e.clean_name
                        if val is None:
                            val = 'None'
                        for c in Dataset.split_sentence(val):
                            self.add_language_token(c)
                with open(f_val_dict, 'wb') as f:
                    d = {}
                    d['_lang_dict'] = Dataset._lang_dict
                    d['_id_lang_map'] = Dataset._id_lang_map
                    cp.dump(d, f, cp.HIGHEST_PROTOCOL)
            print('language dict size', len(Dataset._lang_dict))
        print(Dataset.num_node_types(), 'types of nodes in total.')

    @staticmethod
    def load_type_vocab(f_type_vocab):
        print('loading vocab from', f_type_vocab)
        with open(f_type_vocab, 'rb') as f:
            d = cp.load(f)
            Dataset._node_type_dict = d['_node_type_dict']
            Dataset._id_ntype_map = d['_id_ntype_map']

    @staticmethod
    def split_sentence(sent):
        if Dataset._lang_type == 'word':
            return sent.split()
        elif Dataset._lang_type == 'char':
            return sent
        else:
            raise NotImplementedError

    @staticmethod
    def merge_sentence(sent):
        if Dataset._lang_type == 'word':
            return ' '.join(sent)
        elif Dataset._lang_type == 'char':
            return ''.join(sent)
        else:
            raise NotImplementedError

    def add_language_token(self, token):
        if token in self._lang_dict:
            return
        idx = len(Dataset._lang_dict)
        Dataset._lang_dict[token] = idx
        Dataset._id_lang_map[idx] = token

    def get_sample(self, sname):
        idx = self.sample_index[sname] 
        data_sample = self.data_samples[idx]
        if data_sample.buggy_gnn_graph is None:
            Dataset.setup_gnn_graph(data_sample, self.gnn_type)

        return data_sample

    def load_partition(self):
        self.phase_indices = defaultdict(list)
        for phase in ['train', 'val', 'test']:
            idx_file = os.path.join(self.data_root, '%s.txt' % phase)
            if not os.path.isfile(idx_file):
                continue
            with open(idx_file, 'r') as f:
                for row in f:
                    sname = row.strip()
                    if not sname in self.sample_index:
                        continue

                    self.phase_indices[phase].append(self.sample_index[sname])
            print(phase, 'set has', len(self.phase_indices[phase]), 'samples')

    @staticmethod
    def get_id_of_ntype(node_type):
        return Dataset._node_type_dict[node_type]

    @staticmethod
    def get_value_vocab():
        return Dataset._value_vocab

    @staticmethod
    def set_value_vocab(vocab):
        for key in vocab:
            Dataset.add_value2vocab(key)

    @staticmethod
    def num_tokens():
        return len(Dataset._lang_dict)

    @staticmethod
    def num_const_values():
        return len(Dataset._value_vocab)

    @staticmethod
    def add_value2vocab(val):
        if val in Dataset._value_vocab:
            return
        idx = len(Dataset._value_vocab)
        Dataset._value_vocab[val] = idx
        Dataset._id_value_map[idx] = val

    @staticmethod
    def get_ntype_of_id(idx):
        return Dataset._id_ntype_map[idx]

    @staticmethod
    def num_node_types():
        return len(Dataset._node_type_dict)

    @property
    def num_samples(self):
        return len(self.data_samples)

    @staticmethod
    def _add_node_type(node_type):
        if not node_type in Dataset._node_type_dict:
            idx = len(Dataset._node_type_dict)
            Dataset._node_type_dict[node_type] = idx
            Dataset._id_ntype_map[idx] = node_type

    def build_node_types(self):
        for node_type in shift_node_types: 
            Dataset._add_node_type(node_type)

        for sample in tqdm(self.data_samples):
            for ast in [sample.buggy_ast, sample.fixed_ast]:
                for node in ast.nodes:
                    Dataset._add_node_type(node.node_type)

        for node_type in ADDITIONAL_NODES:
            self._add_node_type(node_type)

    @staticmethod
    def setup_gnn_graph(data_sample, gnn_type):
        #print("setting up", data_sample.buggy_file)
        data_sample.buggy_gnn_graph = get_gnn_graph(data_sample.buggy_code_graph, gnn_type)

    def indices_gen(self, batch_size, phase, infinite):
        if phase == 'train' and infinite and self.resampling:
            idx_per_op = defaultdict(list)
            for idx in self.phase_indices['train']:
                idx_per_op[self.sample_edit_type[idx]].append(idx)
            ops = list(idx_per_op.keys())
            for op in ops:
                print('num', op, len(idx_per_op[op]))
            num_ops = len(ops)
            pos_per_op = [0] * num_ops
            prob_dist = [1.0/num_ops] * num_ops
            base_num = batch_size // num_ops
            rest_num = batch_size - base_num * num_ops
            while True:
                num_per_op = np.random.multinomial(rest_num, prob_dist)
                sub_indices = []
                for i in range(num_ops):
                    num = num_per_op[i] + base_num
                    if pos_per_op[i] + num  > len(idx_per_op[ops[i]]):
                        pos_per_op[i] = 0
                        if num <= len(idx_per_op[ops[i]]):
                            random.shuffle(idx_per_op[ops[i]])
                        else:
                            num = len(idx_per_op[ops[i]])
                    for j in range(pos_per_op[i], pos_per_op[i] + num):
                        sub_indices.append(idx_per_op[ops[i]][j])
                    pos_per_op[i] += num
                yield sub_indices
        else:
            indices = self.phase_indices[phase][:]

            while True:
                if infinite:
                    random.shuffle(indices)
                for i in range(0, len(indices), batch_size):
                    num = min(batch_size, len(indices) - i)
                    if num < batch_size and infinite and len(indices) >= batch_size:
                        break
                    sub_indices = [indices[j] for j in range(i, i + num)]
                    yield sub_indices
                if not infinite:
                    break

    def data_gen(self, batch_size, phase, infinite):
        idx_gen = self.indices_gen(batch_size, phase, infinite)
        
        for sub_indices in idx_gen:
            batch_samples = []
            for j in sub_indices:
                data_sample = self.data_samples[j]
                if data_sample.buggy_gnn_graph is None:
                    Dataset.setup_gnn_graph(data_sample, self.gnn_type)
                for e in data_sample.g_edits:
                    e.clean_unk(data_sample.buggy_code_graph.contents)
                batch_samples.append(data_sample)
            yield batch_samples
            if not self.data_in_mem:
                for sample in batch_samples:
                    sample.unload()

