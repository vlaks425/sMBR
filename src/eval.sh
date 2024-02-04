#!/bin/bash
set -e
# 接收参数
src_file=$1       # 源文件
tgt_file=$2       # 目标文件
output_prefix=$3  # 输出前缀
output_name=$4    # 输出文件名
cuda_devices=$5   # 使用的GPU
# 现有的命令，使用新的变量
sacrebleu /raid/lyu/en-zh/test/zh/test.zho -i $output_prefix/$output_name -m bleu chrf ter -tok zh --confidence > $output_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_devices comet-score -s $src_file -t $output_prefix/$output_name -r $tgt_file  --quiet --only_system >> $output_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_devices python /raid/lyu/evaluate_entity_translation/evaluate_entity_translation.py --ref_file $tgt_file --hypo_file $output_prefix/$output_name --lang zh >> $output_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_devices bert-score -r $tgt_file -c $output_prefix/$output_name --idf --lang zh >> $output_prefix/score.log

CUDA_VISIBLE_DEVICES=$cuda_devices /raid_elmo/home/lr/lyu/conda_env/py39/bin/python -m bleurt.score_files \
  -candidate_file=$output_prefix/$output_name \
  -reference_file=$tgt_file \
  -bleurt_batch_size=100 \
  -batch_same_length=True \
  -bleurt_checkpoint=/cache01/lyu/BLEURT-20 > $output_prefix/bleurt.score

# 平均分数
awk '{sum+=$1} END {print sum/NR}' $output_prefix/bleurt.score >> $output_prefix/score.log
