set -e
phrasing_file=/raid/lyu/QEBT/en2ru/wmt2023.en2ru.mixtral-8x7b-instruct-v0.1.4b-vllm-v0.0.1.json
src_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en
tgt_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru
decoding_method=bs
cuda_devices=0,1
qe_model=Unbabel/wmt22-cometkiwi-da
python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $tgt_file --output_file $tgt_file.filter --phrasing_file $phrasing_file --original_file $src_file --k_input 1
python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $src_file --output_file $src_file.filter --phrasing_file $phrasing_file --original_file $src_file --k_input 1
# 定义一个包含不同 k 值的数组
k_values=(400 340 320 280 240 200 160 128 96 80 64 32 16 5)
#k_values=(280 240 200 160 128 96 80 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/sMBR_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/$decoding_method/$k
  qe_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/qe_$decoding_method/output_$k
  mbr_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/mbr_$decoding_method/output_$k
  mkdir -p $output_prefix
  new_qe_prefix=$output_prefix/qe_baseline
  mkdir -p $new_qe_prefix
  new_baseline_prefix=$output_prefix/baseline
  mkdir -p $new_baseline_prefix
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $baseline_prefix/hypo_$k --output_file $output_prefix/hypo_$k --phrasing_file $phrasing_file --original_file $src_file --k_input $k
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $baseline_prefix/ct2_output.hypo_$k.score --output_file $new_baseline_prefix/ct2_output.hypo_$k.score  --phrasing_file $phrasing_file --original_file $src_file --k_input $k
  awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
  python /home/lr/lyu/QEBT/src/sMBR.py --src_file $phrasing_file --hypo_file $output_prefix/hypo_$k --output_file $output_prefix/phrasing --gpu $cuda_devices --k $k --qe_model $qe_model --src_num 5 --batch_size 64
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $qe_prefix/qe.output.score --output_file $new_qe_prefix/qe.output.score --phrasing_file $phrasing_file --original_file $src_file --k_input $k
  python /home/lr/lyu/QEBT/src/compute_sMBR_score.py --qe_score_file $new_qe_prefix/qe.output.score --phrasing_qe_score_file $output_prefix/phrasing.score --output_file $output_prefix/sMBR.output --hypo_file $output_prefix/hypo_$k --k $k
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $output_prefix/sMBR.output $output_prefix $cuda_devices
  
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $output_prefix/hypo_1 $new_baseline_prefix $cuda_devices
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $qe_prefix/qe.output --output_file $new_qe_prefix/qe.output.filter --phrasing_file $phrasing_file --original_file $src_file --k_input 1
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $new_qe_prefix/qe.output.filter $new_qe_prefix $cuda_devices
  new_mbr_prefix=$output_prefix/mbr_baseline
  mkdir -p $new_mbr_prefix
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $mbr_prefix/mbr.output --output_file $new_mbr_prefix/mbr.output.filter --phrasing_file $phrasing_file --original_file $src_file --k_input 1
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $mbr_prefix/mbr.score --output_file $new_mbr_prefix/mbr.score --phrasing_file $phrasing_file --original_file $src_file --k_input 1
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $new_mbr_prefix/mbr.output.filter $new_mbr_prefix $cuda_devices
  echo "done $k"

done

decoding_method=as
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/sMBR_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/$decoding_method/$k
  qe_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/qe_$decoding_method/output_$k
  mbr_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/mbr_$decoding_method/output_$k
  mkdir -p $output_prefix
  new_qe_prefix=$output_prefix/qe_baseline
  mkdir -p $new_qe_prefix
  new_baseline_prefix=$output_prefix/baseline
  mkdir -p $new_baseline_prefix
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $baseline_prefix/hypo_$k --output_file $output_prefix/hypo_$k --phrasing_file $phrasing_file --original_file $src_file --k_input $k
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $baseline_prefix/ct2_output.hypo_$k.score --output_file $new_baseline_prefix/ct2_output.hypo_$k.score  --phrasing_file $phrasing_file --original_file $src_file --k_input $k
  awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
  python /home/lr/lyu/QEBT/src/sMBR.py --src_file $phrasing_file --hypo_file $output_prefix/hypo_$k --output_file $output_prefix/phrasing --gpu $cuda_devices --k $k --qe_model $qe_model --src_num 5 --batch_size 64
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $qe_prefix/qe.output.score --output_file $new_qe_prefix/qe.output.score --phrasing_file $phrasing_file --original_file $src_file --k_input $k
  python /home/lr/lyu/QEBT/src/compute_sMBR_score.py --qe_score_file $new_qe_prefix/qe.output.score --phrasing_qe_score_file $output_prefix/phrasing.score --output_file $output_prefix/sMBR.output --hypo_file $output_prefix/hypo_$k --k $k
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $output_prefix/sMBR.output $output_prefix $cuda_devices
  
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $output_prefix/hypo_1 $new_baseline_prefix $cuda_devices
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $qe_prefix/qe.output --output_file $new_qe_prefix/qe.output.filter --phrasing_file $phrasing_file --original_file $src_file --k_input 1
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $new_qe_prefix/qe.output.filter $new_qe_prefix $cuda_devices
  new_mbr_prefix=$output_prefix/mbr_baseline
  mkdir -p $new_mbr_prefix
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $mbr_prefix/mbr.output --output_file $new_mbr_prefix/mbr.output.filter --phrasing_file $phrasing_file --original_file $src_file --k_input 1
  python /home/lr/lyu/QEBT/src/filter4sMBR.py --input_file $mbr_prefix/mbr.score --output_file $new_mbr_prefix/mbr.score --phrasing_file $phrasing_file --original_file $src_file --k_input 1
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file.filter $tgt_file.filter $new_mbr_prefix/mbr.output.filter $new_mbr_prefix $cuda_devices
  echo "done $k"

done