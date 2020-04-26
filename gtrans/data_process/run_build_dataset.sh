#!/bin/bash

data_root=[INPUT_DIR]
data_name=[NAME]

save_dir=[OUTPUT_DIR]

python main_build_dataset.py \
    -data_root $data_root \
    -data_name $data_name \
    -save_dir $save_dir \
    -max_ast_nodes 500 \
    -gpu 1 \
    $@

