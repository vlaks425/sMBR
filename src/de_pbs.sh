set -e


#src_file=/raid/lyu/QEBT/en2de/wmt22.en
#tgt_file=/raid/lyu/QEBT/en2de/wmt22a.de
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
cuda_devices=0,1


baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method/5

group_baseline=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method/400

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/mbr_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/qe_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/qe.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/qe.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_mean_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_oracle_mean_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_gpt4_mean_bs/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices


decoding_method=as


baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2de/bs/5

group_baseline=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method/400

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/mbr_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/qe_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/qe.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/qe.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_mean_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_oracle_mean_$decoding_method/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_gpt4_mean_as/output_400
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix de $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/sMBR.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs de $cuda_devices

