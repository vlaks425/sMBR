#!/bin/bash
set -e
# 接收参数
src_file=$1       # 源文件
hypo_file=$2
ref_file=$3
output_prefix=$4
score_file=$5
cuda_devices=$6

python src/create_jsonl4metricx.py --src_file $src_file --hypo_file $hypo_file --ref_file $ref_file --output_file $output_prefix/input.jsonl
CUDA_VISIBLE_DEVICES=$cuda_devices python src/metricx/metricx24/predict.py \
  --tokenizer google/mt5-xxl \
  --model_name_or_path google/metricx-24-hybrid-xxl-v2p6-bfloat16 \
  --max_input_length 1536 \
  --batch_size 4 \
  --input_file $output_prefix/input.jsonl \
  --output_file $output_prefix/metricx24.jsonl

python src/extractor_jsonl4metricx.py --input_file $output_prefix/metricx24.jsonl --output_file $output_prefix/metricx24.scores
rm -rf $output_prefix/input.jsonl
awk '{sum+=$1} END {print sum/NR}' $output_prefix/metricx24.scores >> $output_prefix/$score_file