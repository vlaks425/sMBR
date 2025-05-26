set -e
src_file=data/wmt2023/en-de/generaltest2023.en-de.src.en
tgt_file=data/wmt2023/en-de/generaltest2023.en-de.ref.refA.de
decoding_method=eps
cuda_devices=0
lang_pair=en2de
k_values=(128)
for k in "${k_values[@]}"
do
    echo "start $k"
    output_prefix=output/tower13b_en2de/$decoding_method/$k
    mkdir -p $output_prefix

    if [ $k -le 16 ]; then
        MAX_GEN_NUM_FLAG=""
    else
        MAX_GEN_NUM_FLAG="--max_generation_num 16"
    fi
    
    python src/hf_tower.py --src_file $src_file --output_file $output_prefix/hypo_$k  \
        --num_hypotheses $k --gpu $cuda_devices --lang_pair $lang_pair $MAX_GEN_NUM_FLAG

    awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
    bash src/eval.sh $src_file $tgt_file $output_prefix/hypo_1 $output_prefix $cuda_devices
    echo "done $k"

done
