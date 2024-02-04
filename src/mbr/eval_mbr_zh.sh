set -e
#gpu28
src_file=/raid/lyu/en-zh/test/en/test.en
tgt_file=/raid/lyu/en-zh/test/zh/test.zho
decoding_method=as
cuda_devices=0
base_prefix=/raid/lyu/QEBT/test_dev/test_high_base/$decoding_method
# 定义一个包含不同 k 值的数组
k_values=(320 340 400)
#k_values=(400 340 320 280 240 200 160 128 96 80 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do

    echo "start $k"
    output_prefix=/raid/lyu/QEBT/test_dev/test_high_base/mbr_$decoding_method/output_$k
    mkdir -p $output_prefix
    cp $base_prefix/$k/hypo_$k $output_prefix

    CUDA_VISIBLE_DEVICES=$cuda_devices comet-mbr \
        -s $src_file -t $output_prefix/hypo_$k \
        --num_sample $k -o $output_prefix/mbr.output \
        --output_all_candidates_scores $output_prefix/mbr.score \
        --batch_size 128 \
        --model_storage_path /cache01/lyu/comet_model

    bash /home/lr/lyu/QEBT/src/eval.sh $src_file $tgt_file $output_prefix mbr.output $cuda_devices
    echo "done $k"

done
