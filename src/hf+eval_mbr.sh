set -e

src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
cuda_devices=0
output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/

# 定义一个包含不同 k 值的数组
k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)

for k in "${k_values[@]}"
do
    python /home/lr/lyu/QEBT/src/use_hf_model.py --src_file $src_file --output_file $output_prefix --gpu $cuda_devices --k $k --num_hypotheses $k
    mbr_output_prefix=$output_prefix/mbr_$decoding_method/output_$k
    qe_output_prefix=$output_prefix/qe_$decoding_method/output_$k
    mkdir -p $qe_output_prefix
    mkdir -p $mbr_output_prefix
    cp $output_prefix/$decoding_method/$k/hypo_$k $mbr_output_prefix/hypo_$k
    cp $output_prefix/$decoding_method/$k/hypo_$k $qe_output_prefix/hypo_$k
    awk "NR%$k==1" $output_prefix/$decoding_method/$k/hypo_$k > $output_prefix/$decoding_method/$k/hypo_1
    CUDA_VISIBLE_DEVICES=$cuda_devices comet-mbr \
        -s $src_file -t $mbr_output_prefix/hypo_$k \
        --num_sample $k -o $mbr_output_prefix/mbr.output \
        --output_all_candidates_scores $mbr_output_prefix/mbr.score \
        --batch_size 128 \
        --model_storage_path /cache01/lyu/comet_model
    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $mbr_output_prefix/mbr.output $mbr_output_prefix $cuda_devices
    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $output_prefix/$decoding_method/$k/hypo_1 $output_prefix/$decoding_method/$k $cuda_devices
done
    
