data_name=[NAME]

cooked_root=[DATA_DIR]
save_dir=[MODEL_DIR]

loss_file=[OUTPUT_FILE]
max_num_diffs=1

export CUDA_VISIBLE_DEVICES=1

python find_best_model.py \
               -data_root $cooked_root \
               -data_name $data_name \
               -save_dir $save_dir \
               -gnn_type 's2v_multi' \
               -loss_file $loss_file \
               -max_lv 4 \
               -max_modify_steps $max_num_diffs \
               -resampling True \
               -comp_method mlp \
               $@

