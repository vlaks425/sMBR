set -e

tgt_file=/raid/lyu/QEBT/en2de/wmt2020.en
#src_file=/raid/lyu/QEBT/en2de/wmt2020.de
src_file=/raid/lyu/QEBT/test_dev/test_en2de/bs/5/hypo_1
decoding_method=bs
cuda_devices=0
model_path=/cache01/lyu/meta19/de-en
bpe_codes=/raid_elmo/home/lr/lyu/checkpoints/meta19/de-en/wmt19.de-en.joined-dict.ensemble/bpecodes
bpe_vocab=/raid_elmo/home/lr/lyu/checkpoints/meta19/de-en/wmt19.de-en.joined-dict.ensemble/dict.de.txt
output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/paraphrasing_bt
mkdir -p $output_prefix
# 定义一个包含不同 k 值的数组
#k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)
k=5
python /home/lr/lyu/QEBT/paraphrasing/use_bt.py --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k  \
    --model_path $model_path \
    --beam_size $k --num_hypotheses $k  \
    --batch_size 3000 --use_fastbpe True \
    --bpe_codes $bpe_codes --bpe_vocab $bpe_vocab \
    --src_lang de --tgt_lang en \
    --use_gpu True \
    --gpu $cuda_devices
python /home/lr/lyu/QEBT/src/decode_fastbpe.py --input $output_prefix/ct2_output.hypo_$k --output $output_prefix/hypo_$k --lang en
