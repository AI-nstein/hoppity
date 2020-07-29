import numpy as np
import torch
import re
import os
import sys
import glob
from tqdm import tqdm
from gtrans.common.dataset import Dataset
from gtrans.model.gtrans_model import GraphTrans
from gtrans.common.configs import cmd_args
from gtrans.common.consts import DEVICE

const_val_vocab = np.load(os.path.join(cmd_args.data_root, "vocab_" + cmd_args.vocab_type + ".npy"), allow_pickle=True).item()
Dataset.set_value_vocab(const_val_vocab)
Dataset.add_value2vocab(None)
Dataset.add_value2vocab("UNKNOWN")

dataset = Dataset(cmd_args.data_root, cmd_args.gnn_type)
dataset.load_partition()

torch.set_num_threads(1)

reg = re.escape(cmd_args.save_dir) + r"epoch-([0-9]*).ckpt"
loss_file = cmd_args.loss_file
loss_dict = {}

if not os.path.exists(loss_file):
    open(loss_file, "w").close()
else:
    with open(loss_file, "r") as f:
        c = f.read().split("\n")

    for entry in c:
        key_val = entry.split(":")
        if not len(key_val) == 2:
            continue
        epoch_num = int(key_val[0].strip())
        loss = float(key_val[1].strip())
        loss_dict[epoch_num] = loss

best_loss = None
best_model = None
for _dir in tqdm(glob.glob(os.path.join(cmd_args.save_dir, "*.ckpt"))):
    match = re.match(reg, _dir)
    epoch_num = match.group(1)

    if int(epoch_num) < int(cmd_args.start_epoch) or int(epoch_num) in loss_dict or int(epoch_num) > int(cmd_args.end_epoch):
        continue

    val_gen = dataset.data_gen(cmd_args.batch_size, phase='val', infinite=False)
    model = GraphTrans(cmd_args).to(DEVICE)
    print("loading", _dir)

    model.load_state_dict(torch.load(_dir))
    model.eval()

    tot_samples = 0
    tot_ll = None
    
    for sample_list in tqdm(val_gen):
        ll, new_asts = model(sample_list, phase='val', pred_gt=True)
        tot_ll = torch.sum(ll).item()
        tot_samples += len(sample_list)

    if not tot_ll: 
        print("No validation samples - run split script")
        sys.exit(1)

    test_loss = -tot_ll / tot_samples
    loss_dict[int(epoch_num)] = float(test_loss) 

    with open(loss_file, "a") as f:
        output = epoch_num + ": " + str(test_loss) + "\n"
        f.write(output)

    if best_loss is None or test_loss < best_loss:
        best_model = _dir
        best_loss = test_loss

print("Best Model", best_model)
