set -e

src_file=/raid/lyu/en-zh/test/en/test.en
tgt_file=/raid/lyu/en-zh/test/zh/test.zho
decoding_method=as
cuda_devices=0
model_path=/raid_elmo/home/lr/lyu/checkpoints/en-zh/en2zh/baseline_high_base/model_ct2.pt
spm_path=/raid/lyu/QEBT/en2zh/baseline_high/spm_joint.model
output_prefix=/raid/lyu/QEBT/test_dev/test_high_base/$decoding_method/$k
# 定义一个包含不同 k 值的数组
#k_values=(96 128 160 200 240 280 320 340 400)
k_values=(400 340 320 280 240 200 160 128 96)
# 遍历 k 值
for k in "${k_values[@]}"
do
    output_prefix=/raid/lyu/QEBT/test_dev/test_high_base/$decoding_method/$k
    mkdir -p $output_prefix

    # 执行命令
    #python /home/lr/lyu/QEBT/src/ct2.py --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k --model_path $model_path --spm_path $spm_path --beam_size $k --num_hypotheses $k --gpu $cuda_devices
    python /home/lr/lyu/QEBT/src/ct2_sampling.py  --src_file $src_file --output_file $output_prefix/ct2_output.hypo_$k --model_path $model_path --spm_path $spm_path \
        --beam_size 1 --num_hypotheses $k \
        --use_gpu True --batch_size 300 \
        --gpu $cuda_devices --topk 32001
    python /raid/lyu/fairseq-zh/test.zh.py --input "$output_prefix/ct2_output.hypo_$k" --output "$output_prefix/hypo_$k"
    awk "NR%$k==1" $output_prefix/hypo_$k > $output_prefix/hypo_1
done