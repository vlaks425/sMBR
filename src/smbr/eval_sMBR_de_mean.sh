set -e
phrasing_file=/raid/lyu/QEBT/en2de/wmt2020.en2de.mixtral-8x7b-instruct-v0.1.4b-vllm-v0.0.1.json
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=as
cuda_devices=2
qe_model=Unbabel/wmt22-cometkiwi-da
python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $tgt_file --output_file $tgt_file.filter --phrasing_file $phrasing_file --original_file /raid/lyu/QEBT/en2de/wmt2020.en --k_input 1
python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $src_file --output_file $src_file.filter --phrasing_file $phrasing_file --original_file /raid/lyu/QEBT/en2de/wmt2020.en --k_input 1
# 定义一个包含不同 k 值的数组
#k_values=(5)
k_values=(400)
# 遍历 k 值
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_$decoding_method/output_$k
  new_output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_mean_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method/$k
  mkdir -p $output_prefix
  new_qe_prefix=$output_prefix/qe_baseline
  mkdir -p $new_qe_prefix
  new_baseline_prefix=$output_prefix/baseline
  mkdir -p $new_baseline_prefix
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $baseline_prefix/hypo_$k --output_file $output_prefix/hypo_$k --phrasing_file $phrasing_file --original_file /raid/lyu/QEBT/en2de/wmt2020.en --k_input $k
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $baseline_prefix/ct2_output.hypo_$k.score --output_file $new_baseline_prefix/ct2_output.hypo_$k.score  --phrasing_file $phrasing_file --original_file /raid/lyu/QEBT/en2de/wmt2020.en --k_input $k
  awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
  #python /home/lr/lyu/QEBT/src/sMBR.py --src_file $phrasing_file --hypo_file $output_prefix/hypo_$k --output_file $output_prefix/phrasing --gpu $cuda_devices --k $k --qe_model $qe_model --src_num 5 --batch_size 128
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $qe_prefix/qe.output.score --output_file $new_qe_prefix/qe.output.score --phrasing_file $phrasing_file --original_file /raid/lyu/QEBT/en2de/wmt2020.en --k_input $k --mean
  python /home/lr/lyu/QEBT/src/compute_sMBR_score.py --qe_score_file $new_qe_prefix/qe.output.score --phrasing_qe_score_file $output_prefix/phrasing.score --output_file $output_prefix/sMBR.output --hypo_file $output_prefix/hypo_$k --k $k
  #bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file.filter $tgt_file.filter $output_prefix/sMBR.output $output_prefix $cuda_devices
  bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/mbr.output $output_prefix/hypo_$k $group_baseline $output_prefix/group_pbs de $cuda_devices
  echo "done $k"

done



