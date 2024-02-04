set -e
src_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en
tgt_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru
decoding_method=as
cuda_devices=0,1
qe_model=Unbabel/wmt22-cometkiwi-da
# 定义一个包含不同 k 值的数组
k_values=(400 340 320 280 240 200 160 128 96 80 64 32 16 5)

# 遍历 k 值
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2ru/qe_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2ru/$decoding_method/$k
  mkdir -p $output_prefix
  python /home/lr/lyu/QEBT/src/QE_comet.py --src_file $src_file --hypo_file $baseline_prefix/hypo_$k --output_file $output_prefix/qe.output --gpu $cuda_devices --k $k --qe_model $qe_model --batch_size 128
  echo "done $k"

done

for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=/raid/lyu/QEBT/test_dev/test_en2ru/qe_$decoding_method/output_$k
  baseline_prefix=/raid/lyu/QEBT/test_dev/test_en2ru/$decoding_method/$k
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file $tgt_file $output_prefix/qe.output $output_prefix 0
  bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file $tgt_file $baseline_prefix/hypo_1 $baseline_prefix 0
  echo "done $k"

done