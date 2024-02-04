set -e
cuda_devices=0
# 定义一个包含不同 k 值的数组
k_values=(400 340 320 280 240 200 160 128 96 80 64 32 16 5)
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de.filter
path_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bs
decoding_methods=(bs as)
lang=de
#output_file_name=hypo_1
output_file_name=qe.output
# 遍历 k 值
for k in "${k_values[@]}"
do
    output_prefix=$path_prefix/output_$k/qe_baseline
    CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_number_translation.py --ref_file $tgt_file --hypo_file $output_prefix/qe.output.filter --lang $lang > $output_prefix/num_score.log
    output_prefix=$path_prefix/output_$k/mbr_baseline
    CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_number_translation.py --ref_file $tgt_file --hypo_file $output_prefix/mbr.output.filter --lang $lang > $output_prefix/num_score.log
    output_prefix=$path_prefix/output_$k/baseline
    CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_number_translation.py --ref_file $tgt_file --hypo_file $path_prefix/output_$k/hypo_1 --lang $lang > $output_prefix/num_score.log
    output_prefix=$path_prefix/output_$k
    CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_number_translation.py --ref_file $tgt_file --hypo_file $output_prefix/sMBR.output --lang $lang > $output_prefix/num_score.log
    wait
    echo $k
done