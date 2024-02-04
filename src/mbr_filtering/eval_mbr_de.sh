set -e
#gpu28
#src_file=/raid/lyu/QEBT/en2de/wmt22.en
#tgt_file=/raid/lyu/QEBT/en2de/wmt22a.de
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=as
cuda_devices=0
base_prefix=/raid/lyu/QEBT/test_dev/test_en2de/$decoding_method
# 定义一个包含不同 k 值的数组
k_values=(340 320 280 240 200 160 128 96 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do

    echo "start $k"
    output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/mbr_$decoding_method/filtering2$k/output_$k
    mkdir -p $output_prefix
    awk -v k=$k '{if (NR <= k || (NR % 400) <= k && (NR % 400) != 0) print}' $base_prefix/400/hypo_400 > $output_prefix/hypo_400to$k

    CUDA_VISIBLE_DEVICES=$cuda_devices comet-mbr \
        -s $src_file -t $output_prefix/hypo_400to$k \
        --num_sample $k -o $output_prefix/mbr.output \
        --output_all_candidates_scores $output_prefix/mbr.score \
        --batch_size 128 \
        --model_storage_path /raid/lyu/comet_model \
        --length_batching


    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $output_prefix/mbr.output $output_prefix $cuda_devices

    echo "done $k"

done