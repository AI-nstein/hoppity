#!/bin/bash

save_dir=/home/edinella/test-out/
raw_src=/home/edinella/one-diff-test/01-2019/

python split_train_test.py \
    -save_dir $save_dir \
    -raw_srcs $raw_src \
    $@
