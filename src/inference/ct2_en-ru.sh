set -e
#gpu03 cpu
#src_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en
#tgt_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru
src_file=/raid/lyu/en-ru/tests/Statmt-newstest_enru-2020-eng-rus.eng
tgt_file=/raid/lyu/en-ru/tests/Statmt-newstest_enru-2020-eng-rus.rus
decoding_method=as
cuda_devices=0
model_path=/raid_elmo/home/lr/lyu/meta19_ct2/en-ru
bpe_codes=/raid_elmo/home/lr/lyu/checkpoints/meta19/en-ru/wmt19.en-ru.ensemble/bpecodes
bpe_vocab=/raid_elmo/home/lr/lyu/checkpoints/meta19/en-ru/wmt19.en-ru.ensemble/dict.en.txt
# 定义一个包含不同 k 值的数组
#k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)
k_values=(80)
# 遍历 k 值
for k in "${k_values[@]}"
do

    echo "start $k"
    output_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/$decoding_method/$k
    mkdir -p $output_prefix

    # 执行命令
    #python /home/lr/lyu/QEBT/src/ct2.py --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k --model_path $model_path \
        #--beam_size $k --num_hypotheses $k --gpu $cuda_devices \
        #--batch_size 1 --use_fastbpe True \
        #--bpe_codes $bpe_codes --bpe_vocab $bpe_vocab \
        #--sentence_batch True \
    python /home/lr/lyu/QEBT/src/ct2_sampling.py --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k  \
        --model_path $model_path --use_gpu True \
        --beam_size 1 --num_hypotheses $k  \
        --batch_size 300 --use_fastbpe True \
        --bpe_codes $bpe_codes --bpe_vocab $bpe_vocab \
        --src_lang en --tgt_lang ru \
        --gpu $cuda_devices --topk 31232
    python /home/lr/lyu/QEBT/src/decode_fastbpe.py --input $output_prefix/ct2_output.hypo_$k --output $output_prefix/hypo_$k --lang ru
    awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
    #bash /home/lr/lyu/QEBT/src/eval_ru.sh $src_file $tgt_file $output_prefix/hypo_1 $output_prefix $cuda_devices
    echo "done $k"
done