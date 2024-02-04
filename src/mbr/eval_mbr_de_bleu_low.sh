set -e
#gpu28
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
cuda_devices=2,3
base_prefix=/raid/lyu/QEBT/test_dev/test_en2de_low/$decoding_method
# 定义一个包含不同 k 值的数组
k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)
#k_values=(280 320 340 400)
# 遍历 k 值
for k in "${k_values[@]}"
do

    echo "start $k"
    output_prefix=/raid/lyu/QEBT/test_dev/test_en2de_low/mbr_bleu_$decoding_method/output_$k
    mkdir -p $output_prefix
    cp $base_prefix/$k/hypo_$k $output_prefix
    #python /home/lr/lyu/QEBT/src/bleu_mbr.py --hypo_file $output_prefix/hypo_$k --output_file $output_prefix/mbr.output --k $k --num_workers 26

    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $output_prefix/mbr.output $output_prefix $cuda_devices
    echo "done $k"

done