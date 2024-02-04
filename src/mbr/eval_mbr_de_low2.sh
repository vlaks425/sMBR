set -e
#gpu28
src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
cuda_devices=1
base_prefix=/raid/lyu/QEBT/test_dev/test_en2de_low/$decoding_method
# 定义一个包含不同 k 值的数组
k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)
#k_values=(340 320 280 240 200 160 128 96 80 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do

    echo "start $k"
    output_prefix=/raid/lyu/QEBT/test_dev/test_en2de_low/mbr_$decoding_method/output_$k
    mkdir -p $output_prefix
    cp $base_prefix/$k/hypo_$k $output_prefix

    CUDA_VISIBLE_DEVICES=$cuda_devices comet-mbr \
        -s $src_file -t $output_prefix/hypo_$k \
        --num_sample $k -o $output_prefix/mbr.output \
        --output_all_candidates_scores $output_prefix/mbr.score \
        --batch_size 128 \
        --model_storage_path /cache01/lyu/comet_model

    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $output_prefix/mbr.output $output_prefix $cuda_devices
    echo "done $k"

done
