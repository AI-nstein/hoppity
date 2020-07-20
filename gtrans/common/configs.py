from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import pickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser for graph transform', allow_abbrev=False)
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
cmd_opt.add_argument('-data_root', default=None, help='data_root folder')
cmd_opt.add_argument('-sqr_data', default=None, help='data_root folder for SequenceR')
cmd_opt.add_argument('-data_name', default=None, help='dataset name')
cmd_opt.add_argument('-phase', default=None, help='phase')
cmd_opt.add_argument('-raw_srcs', nargs='*', type=str, help='raw src file folder')

cmd_opt.add_argument('-max_modify_steps', default=10, type=int, help='maximum modifications to the code')
cmd_opt.add_argument('-num_cores', default=4, type=int, help='max num cores used in python')

cmd_opt.add_argument('-data_in_mem', default=False, type=eval, help='keep data in mem?')
cmd_opt.add_argument('-resampling', default=False, type=eval, help='resampling for imbalanced data?')
cmd_opt.add_argument('-loc_given', default=False, type=eval, help='location given during prediction?')
cmd_opt.add_argument('-op_given', default=False, type=eval, help='op given during prediction?')

cmd_opt.add_argument('-max_ast_nodes', default=500, type=int, help='max # nodes in ast')

cmd_opt.add_argument('-ast_fmt', default='shift_node', help='ast format', choices=['gumtree', 'shift_node', 'shift_edge', 'min_node'])
cmd_opt.add_argument('-neg_samples', default=1, type=int, help='# negative sampling')
cmd_opt.add_argument('-rnn_layers', default=2, type=int, help='# rnn layers')
cmd_opt.add_argument('-max_token_len', default=100, type=int, help='max len of name')

cmd_opt.add_argument('-hinge_loss_type', default='sum', help='sum/max')
cmd_opt.add_argument('-lang_dict', default='None', help='None/word/char')

cmd_opt.add_argument('-gnn_msg_dim', default=128, type=int, help='dim of message passing in gnn')
cmd_opt.add_argument('-latent_dim', default=128, type=int, help='latent dim')
cmd_opt.add_argument('-topk', default=1, type=int, help='evaluate topk prediction')
cmd_opt.add_argument('-beam_size', default=1, type=int, help='beam size for beam_search')

cmd_opt.add_argument('-mlp_hidden', default=256, type=int, help='hidden dims in mlp')
cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')

cmd_opt.add_argument('-max_lv', default=3, type=int, help='# layers of gnn')
cmd_opt.add_argument('-msg_agg_type', default='sum', help='how to aggregate the message')
cmd_opt.add_argument('-att_type', default='inner_prod', help='mlp/inner_prod')

cmd_opt.add_argument('-readout_agg_type', default='sum', help='how to aggregate all node embeddings', choices=['sum', 'max', 'mean'])
cmd_opt.add_argument('-gnn_out', default='last', help='how to aggregate readouts from different layers', choices=['last', 'sum', 'max', 'gru', 'mean'])
cmd_opt.add_argument('-gnn_type', default='s2v_code2inv', help='type of graph neural network', choices=['s2v_code2inv', 's2v_single', 's2v_multi'])
cmd_opt.add_argument('-rnn_cell', default='gru', help='type of rnn cell')


cmd_opt.add_argument('-act_func', default='tanh', help='default activation function')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-min_lr', default=1e-6, type=float, help='min learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
cmd_opt.add_argument('-dropout', default=0, type=float, help='dropout')

cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='number of training epochs')
cmd_opt.add_argument('-iters_per_val', default=100, type=int, help='number of iterations per evaluation')
cmd_opt.add_argument('-batch_size', default=10, type=int, help='batch size for training')

cmd_opt.add_argument('-train_pct', default=0.8, type=float, help='fraction of training samples')
cmd_opt.add_argument('-val_pct', default=0.1, type=float, help='fraction of validation samples')
cmd_opt.add_argument('-test_pct', default=0.1, type=float, help='fraction of test samples')

#eval args
cmd_opt.add_argument('-sample_list', nargs='*', type=str, help='list of samples to test on')
cmd_opt.add_argument('-target_model', default=None, type=str, help='path to saved model to test on')
cmd_opt.add_argument('-init_model_dump', default=None, type=str, help='path to saved model to init with')
cmd_opt.add_argument('-loss_file', default="loss.txt", type=str, help='iclr val pred baseline type')
cmd_opt.add_argument('-start_epoch', default=0, type=int, help='lowest number model we should look at')
cmd_opt.add_argument('-end_epoch', default=10000, type=int, help='highest number model we should look at')
cmd_opt.add_argument('-op_breakdown', default=False, type=bool, help='output accuracy breakdown per operation')
cmd_opt.add_argument('-val_acc', default=False, type=bool, help='output accuracy breakdown for value')
cmd_opt.add_argument('-loc_acc', default=False, type=bool, help='output accuracy breakdown for location')
cmd_opt.add_argument('-op_acc', default=False, type=bool, help='output accuracy breakdown for operator')
cmd_opt.add_argument('-type_acc', default=False, type=bool, help='output accuracy breakdown for type')
cmd_opt.add_argument('-sibling_acc', default=False, type=bool, help='output accuracy for chosing child rank when adding a node')
cmd_opt.add_argument('-output_all', default=False, type=bool, help='output correct and incorrect prefixes to files')
cmd_opt.add_argument('-dataset_stats', default=False, type=bool, help='output statistics on the dataset')
cmd_opt.add_argument('-penalize_unknown', default=False, type=bool, help='UNKNOWN predicts will be counted as incorrect')
cmd_opt.add_argument('-rand', default=False, type=bool, help='test the random model')
cmd_opt.add_argument('-beam_agg', default=False, type=bool, help='aggregate all predictions from beam_size=n to 1')
cmd_opt.add_argument('-eval_dump_folder', default="~/eval_dump/", type=str, help='folder where the eval dumps should be saved')

cmd_opt.add_argument('-vocab_type', default="fixes", type=str, help='build vocab with fixes or full', choices=["fixes", "full"])

cmd_opt.add_argument('-comp_method', default="inner_prod", type=str, help='how to evaluate the compatibility method', choices=["inner_prod", "mlp", "bilinear", "multilin"])


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
