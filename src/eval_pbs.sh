#!/bin/bash
set -e
# 接收参数
src_file=$1       # 源文件
tgt_file=$2       # 目标文件
hypo_file=$3      # 预测文件
baseline_hypo_file=$4      # 预测文件
baseline_prefix=$5      # 预测文件
output_prefix=$6  # 输出前缀
lang=$7          # 语言
cuda_devices=$8   # 使用的GPU
# 现有的命令，使用新的变量
mkdir -p $output_prefix
sacrebleu $tgt_file -i $baseline_hypo_file $hypo_file -m bleu chrf ter --paired-bs > $output_prefix/score_pbs.log
CUDA_VISIBLE_DEVICES=$cuda_devices comet-compare -s $src_file -t $baseline_hypo_file $hypo_file -r $tgt_file  --quiet --only_system >> $output_prefix/score_pbs.log
if [ ! -f "$output_prefix/bleurt.score" ]; then
    CUDA_VISIBLE_DEVICES=$cuda_devices /raid_elmo/home/lr/lyu/conda_env/py39/bin/python -m bleurt.score_files \
      -candidate_file=$hypo_file \
      -reference_file=$tgt_file \
      -bleurt_batch_size=100 \
      -batch_same_length=True \
      -bleurt_checkpoint=/raid/lyu/BLEURT-20 > $output_prefix/bleurt.score
fi
#$baseline_prefix/bleurt.score不存在的话，先生成
if [ ! -f "$baseline_prefix/bleurt.score" ]; then
    CUDA_VISIBLE_DEVICES=$cuda_devices /raid_elmo/home/lr/lyu/conda_env/py39/bin/python -m bleurt.score_files \
      -candidate_file=$baseline_hypo_file \
      -reference_file=$tgt_file \
      -bleurt_batch_size=100 \
      -batch_same_length=True \
      -bleurt_checkpoint=/raid/lyu/BLEURT-20 > $baseline_prefix/bleurt.score
fi

python /home/lr/lyu/QEBT/src/bleurt_pbs.py --hypo_files_scores $baseline_prefix/bleurt.score $output_prefix/bleurt.score >> $output_prefix/score_pbs.log
