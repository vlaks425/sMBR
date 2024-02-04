set -e
src_num=5
phrasing_file=/raid/lyu/QEBT/test_dev/test_en2de/paraphrasing_bt/hypo_$src_num.json
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
cuda_devices=0,1
qe_model=Unbabel/wmt22-cometkiwi-da
# 定义一个包含不同 k 值的数组
#k_values=(400)
k_values=(340 320 280 240 200 160 128 96 80 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_$decoding_method/output_$k
  new_output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_mean_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method/$k
  qe_prefix=/raid/lyu/QEBT/test_dev/test_en2de/qe_$decoding_method/output_$k
  mbr_prefix=/raid/lyu/QEBT/test_dev/test_en2de/mbr_$decoding_method/output_$k
  mkdir -p $new_output_prefix
  #python /home/lr/lyu/QEBT/src/sMBR.py --src_file $phrasing_file --hypo_file $baseline_prefix/hypo_$k --output_file $output_prefix/phrasing --gpu $cuda_devices --k $k --qe_model $qe_model --src_num $src_num --batch_size 64
  python /home/lr/lyu/QEBT/src/compute_sMBR_score.py --qe_score_file $qe_prefix/qe.output.score  \
    --phrasing_qe_score_file $output_prefix/phrasing.score --output_file $new_output_prefix/sMBR.output --hypo_file $baseline_prefix/hypo_$k  \
    --k $k --mean
  #bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $new_output_prefix/sMBR.output $new_output_prefix $cuda_devices
  bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $new_output_prefix de $cuda_devices
  #bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/sMBR.output $mbr_prefix/mbr.output $mbr_prefix $new_output_prefix/with_mbr de $cuda_devices
  #bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/sMBR.output $qe_prefix/qe.output $qe_prefix $new_output_prefix/with_qe de $cuda_devices
  echo "done $k"

done
decoding_method=as
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_$decoding_method/output_$k
  new_output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/sMBR_bt_mean_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method/$k
  qe_prefix=/raid/lyu/QEBT/test_dev/test_en2de/qe_$decoding_method/output_$k
  mbr_prefix=/raid/lyu/QEBT/test_dev/test_en2de/mbr_$decoding_method/output_$k
  mkdir -p $new_output_prefix
  #python /home/lr/lyu/QEBT/src/sMBR.py --src_file $phrasing_file --hypo_file $baseline_prefix/hypo_$k --output_file $output_prefix/phrasing --gpu $cuda_devices --k $k --qe_model $qe_model --src_num $src_num --batch_size 64
  python /home/lr/lyu/QEBT/src/compute_sMBR_score.py --qe_score_file $qe_prefix/qe.output.score  \
    --phrasing_qe_score_file $output_prefix/phrasing.score --output_file $new_output_prefix/sMBR.output --hypo_file $baseline_prefix/hypo_$k  \
    --k $k --mean
  #bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $new_output_prefix/sMBR.output $new_output_prefix $cuda_devices
  bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/sMBR.output $baseline_prefix/hypo_1 $baseline_prefix $new_output_prefix de $cuda_devices
  #bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/sMBR.output $mbr_prefix/mbr.output $mbr_prefix $new_output_prefix/with_mbr de $cuda_devices
  #bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $new_output_prefix/sMBR.output $qe_prefix/qe.output $qe_prefix $new_output_prefix/with_qe de $cuda_devices
  echo "done $k"

done
