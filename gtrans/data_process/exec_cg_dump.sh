cooked_root=~/cooked-full-fmt-shift_node
data_name=full

save_dir=~/scratch/hoppity_cg

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


python dump_cg.py \
	-data_root $cooked_root \
	-data_name $data_name \
	-save_dir $save_dir \
	-gnn_type 's2v_multi' \
	-max_lv 4 \
	-max_modify_steps 1 \
	-resampling True \
	-comp_methd "mlp" \
	$@
