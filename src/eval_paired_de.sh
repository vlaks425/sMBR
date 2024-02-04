set -e
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
lang=de
cuda_devices=0,1
qe_model=Unbabel/wmt22-cometkiwi-da
path_prefix=/raid/lyu/QEBT/test_dev/test_en2de
# 定义一个包含不同 k 值的数组
k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)
# 遍历 k 值
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=$path_prefix/sMBR_$decoding_method/output_$k
  baseline_prefix=$path_prefix/$decoding_method/$k
  qe_prefix=$path_prefix/qe_$decoding_method/output_$k
  mbr_prefix=$path_prefix/mbr_$decoding_method/output_$k
  new_qe_prefix=$output_prefix/qe_baseline
  new_baseline_prefix=$output_prefix/baseline
  new_mbr_prefix=$output_prefix/mbr_baseline

  CUDA_VISIBLE_DEVICES=$cuda_devices comet-compare -s $src_file.filter -r $tgt_file.filter -t $output_prefix/hypo_1 $new_qe_prefix/qe.output.filter $new_mbr_prefix/mbr.output.filter $output_prefix/sMBR.output > $output_prefix/score_ps.log
  CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_entity_translation_pbs.py --ref_file $tgt_file.filter --hypo_files $output_prefix/hypo_1 $new_qe_prefix/qe.output.filter $new_mbr_prefix/mbr.output.filter $output_prefix/sMBR.output --lang $lang >> $output_prefix/score_ps.log
  CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/bertscore.py --ref_file $tgt_file.filter --hypo_files $output_prefix/hypo_1 $new_qe_prefix/qe.output.filter $new_mbr_prefix/mbr.output.filter $output_prefix/sMBR.output --lang $lang >> $output_prefix/score_ps.log
  
  echo "done $k"

done


decoding_method=as
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=$path_prefix/sMBR_$decoding_method/output_$k
  baseline_prefix=$path_prefix/$decoding_method/$k
  qe_prefix=$path_prefix/qe_$decoding_method/output_$k
  mbr_prefix=$path_prefix/mbr_$decoding_method/output_$k
  new_qe_prefix=$output_prefix/qe_baseline
  new_baseline_prefix=$output_prefix/baseline
  new_mbr_prefix=$output_prefix/mbr_baseline

  CUDA_VISIBLE_DEVICES=$cuda_devices comet-compare -s $src_file.filter -r $tgt_file.filter -t $output_prefix/hypo_1 $new_qe_prefix/qe.output.filter $new_mbr_prefix/mbr.output.filter $output_prefix/sMBR.output > $output_prefix/score_ps.log
  CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_entity_translation_pbs.py --ref_file $tgt_file.filter --hypo_files $output_prefix/hypo_1 $new_qe_prefix/qe.output.filter $new_mbr_prefix/mbr.output.filter $output_prefix/sMBR.output --lang $lang >> $output_prefix/score_ps.log
  CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/bertscore.py --ref_file $tgt_file.filter --hypo_files $output_prefix/hypo_1 $new_qe_prefix/qe.output.filter $new_mbr_prefix/mbr.output.filter $output_prefix/sMBR.output --lang $lang >> $output_prefix/score_ps.log
  
  echo "done $k"

done