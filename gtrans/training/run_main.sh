cooked_root=[INPUT_DIR]
data_name=[NAME]

save_dir=[OUTPUT_DIR]

python main_gtrans.py \
	-data_root $cooked_root \
	-data_name $data_name \
	-save_dir $save_dir \
	-gnn_type 's2v_multi' \
	-max_lv 4 \
	-max_modify_steps 1 \
	-resampling True \
	-comp_method "mlp" \
	$@
