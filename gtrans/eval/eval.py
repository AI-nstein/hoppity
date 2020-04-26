import os
import numpy as np
from time import time
import torch
from tqdm import tqdm
from gtrans.eval.utils import ast_acc_cnt, setup_dicts, loc_acc_cnt, val_acc_cnt, type_acc_cnt, op_acc_cnt, get_top_k, get_val
from gtrans.data_process.utils import get_bug_prefix
from gtrans.common.configs import cmd_args
from gtrans.common.dataset import Dataset, GraphEditCmd
from gtrans.model.gtrans_model import GraphTrans
from gtrans.common.consts import DEVICE
from gtrans.common.consts import OP_REPLACE_VAL, OP_ADD_NODE, OP_REPLACE_TYPE, OP_DEL_NODE, OP_NONE

const_val_vocab = np.load(os.path.join(cmd_args.data_root, "vocab_" + cmd_args.vocab_type + ".npy"), allow_pickle=True).item()
Dataset.set_value_vocab(const_val_vocab)
Dataset.add_value2vocab(None)
Dataset.add_value2vocab("UNKNOWN")

dataset = Dataset(cmd_args.data_root, cmd_args.gnn_type)
dataset.load_partition()

phase = "test"

torch.set_num_threads(1)

def sample_gen(s_list):
    yield s_list

#either input a list of inputs or just generate some from the test set 
if not cmd_args.sample_list:
    val_gen = dataset.data_gen(cmd_args.batch_size, phase=phase, infinite=False)
else:
    new_sample_list = []
    for sample in cmd_args.sample_list:
        new_sample_list.append(dataset.get_sample(sample)) 

    val_gen = sample_gen(new_sample_list)

model = GraphTrans(cmd_args).to(DEVICE)
print("loading", cmd_args.target_model)

if cmd_args.rand:
    model.set_rand_flag(True)

model.load_state_dict(torch.load(cmd_args.target_model))
model.eval()

op_acc, loc_acc_op, val_acc_op, type_acc_op, true_ops, cor_ops = setup_dicts()
total_num_samples = 0
total_acc, total_loc_acc, total_val_acc, total_type_acc, total_op_acc = 0, 0, 0, 0, 0

_t =0
count = 0
type_count = 0
val_count = 0
loc_count = 0
unk_count = 0

bug_dict = {}

unique_edit = 0

if cmd_args.output_all:
    open("cor_prefixes.txt", "w").close()
    open("wrong_prefixes.txt", "w").close()

print("Beam agg", cmd_args.beam_agg)

for sample_list in tqdm(val_gen):

    total_num_samples += len(sample_list)

    ll_total = [[] for i in range(len(sample_list))]
    new_asts_total = [[] for i in range(len(sample_list))]

    if cmd_args.beam_agg:
        for b in range(1, cmd_args.beam_size+1):
            ll, new_asts = model(sample_list, phase='test', beam_size=b, pred_gt=False, op_given=cmd_args.op_given, loc_given=cmd_args.loc_given)
           
            b_edits = new_asts[0]

            assert len(ll) == b * len(sample_list)

            for i in range(0, len(sample_list)):
                start_idx = i*b
                end_idx = start_idx + b

                s_ll = ll[start_idx:end_idx].tolist()
                ll_total[i] += s_ll

                s_asts = new_asts[i]
                new_asts_total[i] += s_asts

        ll, new_asts = get_top_k(ll_total, new_asts_total, len(sample_list), cmd_args.beam_size)
    else:
        ll, new_asts = model(sample_list, phase='test', beam_size=cmd_args.beam_size, pred_gt=False, op_given=cmd_args.op_given, loc_given=cmd_args.loc_given)

    contents = [s.buggy_code_graph.contents.keys() | Dataset._value_vocab.keys() for s in sample_list]

    #dataset_stats
    if cmd_args.max_modify_steps == 1:
        sample_true_ops = [s.g_edits[0].op for s in sample_list]
        for op in sample_true_ops:
            true_ops[op] += 1
    else:
        sample_true_ops = [len(s.g_edits) for s in sample_list]
        for op in sample_true_ops:
            true_ops[op] += 1

    num_nodes = [s.fixed_ast.num_nodes for s in sample_list]
    acc_lst = ast_acc_cnt(new_asts, [s.fixed_ast for s in sample_list], contents)
    
    if cmd_args.max_modify_steps == 1:
        ops = [s.g_edits[0].op for (acc, s) in zip(acc_lst, sample_list) if acc]
        for op in ops:
            cor_ops[op] += 1
    else:
        ops = [len(s.g_edits) for (acc, s) in zip(acc_lst, sample_list) if acc]

        for op in ops:
            cor_ops[op] += 1

    acc = sum(acc_lst)
    total_acc += acc

    cor_prefixes = [s.f_bug for (acc, s) in zip(acc_lst, sample_list) if acc]
    wrong_prefixes = [s.f_bug for (acc, s) in zip(acc_lst, sample_list) if not acc]

    if cmd_args.output_all:
        out_str = ""
        for prefix in cor_prefixes:
            out_str += str(prefix) + "\n"

        w_out_str = ""
        for prefix in wrong_prefixes:
            w_out_str += str(prefix) + "\n"

        with open("cor_prefixes.txt", "a") as f:
            f.write(out_str)

        with open("wrong_prefixes.txt", "a") as f:
            f.write(w_out_str)

    
    new_edits = []
    for ast in new_asts:
        ast_edits = [x.get_edits() if x.get_edits() else [GraphEditCmd("NoOp")] for x in ast[:cmd_args.topk]]
        new_edits.append(ast_edits)

    true_edits = [s.g_edits for s in sample_list]
    

    total_op_acc += op_acc_cnt(new_edits, true_edits)
    total_loc_acc += loc_acc_cnt(new_edits, true_edits)
    loc_count += 1

    for edit in true_edits:
        for e in edit:
            if hasattr(e, "node_name") and e.node_name == "UNKNOWN":
                unk_count += 1

    val_acc = val_acc_cnt(new_edits, true_edits, contents)
    if val_acc >= 0:
        val_count += 1
        total_val_acc += val_acc

    type_acc = type_acc_cnt(new_edits, true_edits)
    #print(new_edits, true_edits, type_acc)

    '''
    for e_idx, e in enumerate(true_edits):
        for s_idx, step in enumerate(e):
            if step.op == OP_REPLACE_TYPE or step.op == OP_ADD_NODE:
                true_type = step.node_type

                p_type = new_edits[e_idx][s_idx][0].node_type if len(new_edits[e_idx]) > s_idx and (new_edits[e_idx][s_idx][0].op == OP_REPLACE_TYPE or new_edits[e_idx][s_idx][0].op == OP_ADD_NODE) else None

                if true_type == p_type:
                    _t += 1
                
                type_count += 1
     '''

    if type_acc >= 0:
        total_type_acc += type_acc
        type_count += 1

    if cmd_args.max_modify_steps == 1:
        for OP in [OP_ADD_NODE, OP_REPLACE_VAL, OP_REPLACE_TYPE, OP_DEL_NODE, OP_NONE]:
            idxs = [ i for i in range(0, len(sample_list)) if sample_list[i].g_edits[0].op == OP ]
            pred_ops = [ new_edits[i][0][0].op if new_edits[i][0]  else OP_NONE for i in idxs ]

            op_acc[OP]["total_op"] += len(idxs)

            for OP2 in [OP_ADD_NODE, OP_REPLACE_VAL, OP_REPLACE_TYPE, OP_DEL_NODE, OP_NONE]:
                op_acc[OP][OP2] += len([ op for op in pred_ops if op == OP2 ])

            loc_acc_idxs = [ loc_acc_cnt([new_edits[i]], [sample_list[i].g_edits]) for i in idxs ]
            loc_acc_op[OP] += sum(loc_acc_idxs)

            if OP in type_acc_op:
                type_acc_op[OP] += sum([ type_acc_cnt([new_edits[i]], [sample_list[i].g_edits]) for i in idxs ])

            if OP in val_acc_op:
                val_acc_op[OP] += sum([ val_acc_cnt([new_edits[i]], [sample_list[i].g_edits], contents) for i in idxs ])

    else:
        for i in range(0, cmd_args.max_modify_steps+1):
            idxs = [ i for i in range(0, len(sample_list)) if len(sample_list[i].g_edits) == i ]
            pred_ops = [len(new_edits[i][0]) for i in idxs]
            op_acc[i]["total_op"] += len(idxs)

            for j in range(0, cmd_args.max_modify_steps+1):
                op_acc[i][j] += len([ op for op in pred_ops if op == j])

            loc_acc_idxs = [ loc_acc_cnt([new_edits[i]], [sample_list[i].g_edits]) for i in idxs ]
            loc_acc_op[i] += sum(loc_acc_idxs)

            #type_acc_op[i] += sum([ type_acc_cnt([new_edits[i]], [sample_list[i].g_edits]) for i in idxs ])
            #val_acc_op[i] += sum([ val_acc_cnt([new_edits[i]], [sample_list[i].g_edits], contents) for i in idxs ])

    count += 1

print('total accuracy %.4f\n' % (total_acc / total_num_samples))

final_loc_acc = total_loc_acc / loc_count
if cmd_args.loc_acc:
    print('location accuracy %.4f' % (final_loc_acc))

final_op_acc = total_op_acc / total_num_samples
if cmd_args.op_acc:
    print('op accuracy %.4f' % (final_op_acc))

final_val_acc = total_val_acc / val_count
if cmd_args.val_acc:
    print('value accuracy %.4f' % (final_val_acc))

final_type_acc = total_type_acc / type_count if type_count > 0 else 0
if cmd_args.type_acc:
    print('type accuracy %.4f' % (final_type_acc))


if cmd_args.op_breakdown:
    if cmd_args.max_modify_steps == 1:
        op_lst = [OP_ADD_NODE, OP_REPLACE_VAL, OP_REPLACE_TYPE, OP_DEL_NODE, OP_NONE]
    else:
        op_lst = range(0, cmd_args.max_modify_steps+1)

    for op in op_lst:
        print("\n" + str(op), "accuracy")
        for k, v in op_acc[op].items():
            print(k, v)

        if loc_acc_op[op] == 0 or true_ops[op] == 0:
            print()
            continue

        print("loc acc %.4f, total: %d" % ((loc_acc_op[op] / true_ops[op]), loc_acc_op[op]))

        if op in val_acc_op:
            print("val acc %.4f, total: %d"  % ((val_acc_op[op] / true_ops[op]), val_acc_op[op]))

        if op in type_acc_op:
            print("type acc %.4f, total: %d" % ((type_acc_op[op] / true_ops[op]), type_acc_op[op]))
        print()

if cmd_args.dataset_stats:
    print("\ntotal true op dataset statistics")
    for k, v in true_ops.items():
        print(k, v)

fname = str(time()).replace(".", "_")
f_args = os.path.expanduser(os.path.join(cmd_args.eval_dump_folder, fname + "_args.npy"))
np.save(f_args, cmd_args)

output_dict = {}
output_dict["phase"] = phase
output_dict["val_acc"] = final_val_acc
output_dict["type_acc"] = final_type_acc
output_dict["loc_acc"] = final_loc_acc
output_dict["op_acc"] = final_op_acc
output_dict["op_breakdown"] = op_acc
output_dict["true_ops"] = true_ops

f_output_dict = os.path.expanduser(os.path.join(cmd_args.eval_dump_folder, fname + "_dump.npy"))
np.save(f_output_dict, output_dict)


print("number of unique edits", unique_edit, "total samples", total_num_samples)
print("number of unknowns", unk_count)
