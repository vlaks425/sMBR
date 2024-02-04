set -e
#gpu28
#src_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en
#tgt_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru
src_file=/raid/lyu/en-ru/tests/Statmt-newstest_enru-2020-eng-rus.eng
tgt_file=/raid/lyu/en-ru/tests/Statmt-newstest_enru-2020-eng-rus.rus
decoding_method=bs
cuda_devices=1
base_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/$decoding_method
# 定义一个包含不同 k 值的数组
k_values=(80 80 80)
#k_values=(400 340 320 280 240 200 160 128 96 80 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do

    echo "start $k"
    output_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/mbr_$decoding_method/output_$k
    #new_output_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/mbr_$decoding_method/output_qe_reranking_80_$k
    mkdir -p $output_prefix
    #mkdir -p $new_output_prefix
    cp $base_prefix/$k/hypo_$k $output_prefix

    CUDA_VISIBLE_DEVICES=$cuda_devices comet-mbr \
        -s $src_file -t $output_prefix/hypo_$k \
        --num_sample $k -o $output_prefix/mbr.output \
        --output_all_candidates_scores $output_prefix/mbr.score \
        --batch_size 128 \
        --model_storage_path /local/lyu/comet_model

    #bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file $tgt_file $output_prefix/mbr.output $output_prefix $cuda_devices
    echo "done $k"

done
bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file $tgt_file $output_prefix/mbr.output $output_prefix $cuda_devices
#bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file $tgt_file $new_output_prefix/mbr.output $new_output_prefix $cuda_devices