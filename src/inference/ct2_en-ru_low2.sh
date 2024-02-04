set -e
#cpu03
src_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en
tgt_file=/raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru
decoding_method=bs
cuda_devices=1
model_path=/cache01/lyu/checkpoints/en-ru/low
spm_path=/raid/lyu/fairseq-ru/baseline/spm_joint.model
# 定义一个包含不同 k 值的数组
#k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 400)
k_values=(200 160 128 96 80 64 32 16 5)
# 遍历 k 值
for k in "${k_values[@]}"
do
    output_prefix=/raid/lyu/QEBT/test_dev/test_en2ru_low/$decoding_method/$k
    mkdir -p $output_prefix

    # 执行命令
    python /home/lr/lyu/QEBT/src/ct2.py --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k  \
        --model_path $model_path --beam_size $k  \
        --spm_path $spm_path \
        --batch_size 3000 \
        --num_hypotheses $k --gpu $cuda_devices
    #python /home/lr/lyu/QEBT/src/ct2_sampling.py --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k  \
        #--model_path $model_path --spm_path $spm_path  --use_gpu True \
        #--beam_size 1 --num_hypotheses $k  \
        #--batch_size 300 \
        #--gpu $cuda_devices --topk 32001
    cp $output_prefix/ct2_output.hypo_$k $output_prefix/hypo_$k
    awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
    echo "done $k"
done