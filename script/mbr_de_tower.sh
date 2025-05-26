set -e
src_file=data/wmt2023/en-de/generaltest2023.en-de.src.en
tgt_file=data/wmt2023/en-de/generaltest2023.en-de.ref.refA.de
cuda_devices=0
decoding_method=eps
k_values=(128 5)
for k in "${k_values[@]}"
do
    base_prefix=output/tower13b_en2de/$decoding_method
    echo "start $k"
    output_prefix=output/tower13b_en2de/mbr_$decoding_method/$k
    mkdir -p $output_prefix

    CUDA_VISIBLE_DEVICES=$cuda_devices comet-mbr \
        -s $src_file -t $base_prefix/$k/hypo_$k \
        --num_sample $k -o $output_prefix/mbr.output \
        --batch_size 256 

    bash src/eval.sh $src_file $tgt_file $output_prefix/mbr.output $output_prefix $cuda_devices
    echo "done $k"

done