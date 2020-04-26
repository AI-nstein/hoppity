import copy
from gtrans.common.consts import OP_REPLACE_VAL, OP_ADD_NODE, OP_REPLACE_TYPE, OP_DEL_NODE, OP_NONE, SEPARATOR as SEP
from gtrans.common.dataset import GraphEditCmd
from gtrans.common.configs import cmd_args
from gtrans.common.code_graph import tree_equal, eq_val

op_names = [OP_NONE, OP_REPLACE_VAL, OP_DEL_NODE, OP_REPLACE_TYPE, OP_ADD_NODE]
accs = {
        OP_ADD_NODE : 0,
        OP_REPLACE_VAL : 0,
        OP_REPLACE_TYPE : 0,
        OP_DEL_NODE : 0,
        OP_NONE: 0,
        "total_op": 0
}

accs_2 = {}
for i in range(0, cmd_args.max_modify_steps+1):
    accs_2[i] = 0

accs_2["total_op"] = 0

def get_clean_type(node, idx):
    if SEP in node.node_type:
        return node.node_type.split(SEP)[idx]
    else:
        return node.node_type

def setup_dicts():
    op_acc, loc_acc, val_acc, type_acc, true_ops, cor_ops = None, None, None, None, None, None

    op_acc = {}
    if cmd_args.max_modify_steps == 1:
        for op in op_names:
            op_acc[op] = copy.deepcopy(accs)
    else:
        for i in range(0, cmd_args.max_modify_steps+1):
            op_acc[i] = copy.deepcopy(accs_2)

    if cmd_args.max_modify_steps == 1:
        loc_acc = copy.deepcopy(accs)
    else:
        loc_acc = {}
        for i in range(0, cmd_args.max_modify_steps+1):
            loc_acc[i] = 0

        loc_acc["total_op"] = 0


    val_acc = {
        OP_ADD_NODE : 0,
        OP_REPLACE_VAL : 0
    }

    type_acc = {
        OP_ADD_NODE : 0,
        OP_REPLACE_TYPE : 0,
    }


    true_ops = {}
    if cmd_args.max_modify_steps == 1:
        for op in op_names:
            true_ops[op] = 0
    else:
        for i in range(0, cmd_args.max_modify_steps+1):
            true_ops[i] = 0

    if cmd_args.max_modify_steps == 1:
        cor_ops = copy.deepcopy(accs)
    else:
        cor_ops = {}
        for i in range(0, cmd_args.max_modify_steps+1):
            cor_ops[i] = 0

    return op_acc, loc_acc, val_acc, type_acc, true_ops, cor_ops

def ast_acc_cnt(pred_asts, true_asts, contents):
    acc = [0] * len(pred_asts)
    assert len(pred_asts) == len(true_asts)
    idx = 0
    for x_list, y, c in zip(pred_asts, true_asts, contents):
        x_list = x_list[:cmd_args.topk]
        for x in x_list:
            if tree_equal(x, y, c):
                acc[idx] = 1
                break

        idx += 1

    return acc

def acc_cnt(pred_asts, true_asts, eq_func, contents=None):
    acc = 0
    assert len(pred_asts) == len(true_asts)

    i = 0
    for x_list, y in zip(pred_asts, true_asts):
        mismatch = False
        for (e_idx, y_edit) in enumerate(y):

            step_match = False
            for x in x_list:
                if len(x) <= e_idx:
                    continue

                x = x[e_idx]
                if (not contents and eq_func(x, y_edit)) or (contents and eq_func(x, y_edit, contents[i])):
                    step_match = True
                    break

            if not step_match:
                mismatch = True
                break

        if not mismatch:
            acc += 1

        i += 1

    return acc


def get_loc(graph_edit):
    if graph_edit.op == OP_ADD_NODE:
        return (graph_edit.parent_id, graph_edit.child_rank)
    else:
        return graph_edit.node_index

def get_type(graph_edit):
    if graph_edit.op == OP_ADD_NODE or graph_edit.op == OP_REPLACE_TYPE:
        return graph_edit.node_type
    else:
        return None

def get_val(graph_edit):
    if graph_edit.op == OP_ADD_NODE or graph_edit.op == OP_REPLACE_VAL:
        return graph_edit.node_name
    else:
        return -1

def loc_acc_cnt(pred_asts, true_asts):
    p_asts, t_asts = filter_by_op(pred_asts, true_asts, [OP_ADD_NODE, OP_REPLACE_VAL, OP_REPLACE_TYPE, OP_DEL_NODE])
    acc = acc_cnt(p_asts, t_asts, lambda x, y: get_loc(x) == get_loc(y))

    return -1 if len(t_asts) == 0 else acc / len(t_asts)

def op_acc_cnt(pred_asts, true_asts):

    p_asts = []
    for p in pred_asts:
        p_asts.append([p[0]])

    acc = acc_cnt(p_asts, true_asts, lambda x, y: x.op == y.op)
    return acc

def filter_by_op(pred_asts, true_asts, ops_allowed):
    p_asts = []
    t_asts = []
    for (t_idx, t) in enumerate(true_asts):
        p = pred_asts[t_idx]

        if t[0].op == OP_NONE:
            continue

        p_s = []
        for k in p:
            p_s.append([])

        t_s = []
        for (s_idx, step) in enumerate(t):

            if step.op in ops_allowed:
                t_s.append(step)

                for (k_idx, k) in enumerate(p):
                    if len(k) <= s_idx:
                        p_s[k_idx].append(GraphEditCmd("NoOp"))
                    else:
                        p_s[k_idx].append(k[s_idx])

        assert len(p_s) == len(p)

        if len(t_s) > 0:
            t_asts.append(t_s)
            p_asts.append(p_s)

    return p_asts, t_asts

def type_acc_cnt(pred_asts, true_asts):
    p_asts, t_asts = filter_by_op(pred_asts, true_asts, [OP_ADD_NODE, OP_REPLACE_TYPE])
    total_type_ops = len(t_asts)

    acc = acc_cnt(p_asts, t_asts, lambda x, y: get_type(y) and get_type(x) == get_type(y))
    return -1 if total_type_ops == 0 else acc / total_type_ops

def val_acc_cnt(pred_asts, true_asts, contents):
    p_asts, t_asts = filter_by_op(pred_asts, true_asts, [OP_ADD_NODE, OP_REPLACE_VAL])

    total_val_ops = len(t_asts)

    acc = acc_cnt(p_asts, t_asts, lambda x, y, c: eq_val(get_val(x), get_val(y), c), contents)

    return -1 if total_val_ops == 0 else acc / total_val_ops

def sib_acc_cnt(pred_asts, true_asts):
    total_adds = len([y for edits in true_asts for y in edits if y.op == OP_ADD_NODE])
    return -1 if total_adds == 0 else acc_cnt(pred_asts, true_asts, lambda x, y: y.op == OP_ADD_NODE and x.op == OP_ADD_NODE and y.parent_id == x.parent_id and x.child_rank == y.child_rank) / total_adds

def _remove_dups(lst, ll):
    edits = [a.get_edits()[0] for a in lst]
    while True:
        changed = False
        for i in range(0, len(edits)):
            ast = lst[i]
            loss = ll[i]
            edit = edits[i]
            if edits.count(edit) > 1:
                lst.remove(ast)
                edits.remove(edit)
                ll.remove(loss)

                changed = True
                break

        if not changed:
            return lst, ll

def get_top_k(ll_total, new_asts_total, tot_samples, beam_size):
    ll = [[] for i in range(tot_samples)]
    new_asts = [[] for i in range(tot_samples)]

    for (i, elem) in enumerate(tot_samples):
        s_ll = ll_total[i]
        s_asts = new_asts_total[i]

        s_asts, s_ll = _remove_dups(s_asts, s_ll)

        for k in range(0, beam_size):
            try:
                s_max = max(s_ll)
            except ValueError:
                break

            idx = s_ll.index(s_max)
            s_ll.remove(s_max)

            elem = s_asts[idx]
            s_asts.remove(elem)

            ll[i].append(s_max)
            new_asts[i].append(elem)

    return ll, new_asts
