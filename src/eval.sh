#!/bin/bash
set -e

src_file=$1
tgt_file=$2
hypo_file=$3
output_prefix=$4
cuda_devices=$5

sacrebleu $tgt_file -i $hypo_file -m bleu --confidence > $output_prefix/score.log

CUDA_VISIBLE_DEVICES=$cuda_devices comet-score -s $src_file -t $hypo_file -r $tgt_file --only_system --model Unbabel/XCOMET-XXL --model_storage_path /home/2/uh02312/lyu_fs/cache --batch_size 1 > $output_prefix/score_xxl_24.log

# MetricX-24
bash /home/2/uh02312/lyu/QP_problem/src/eval_metricx_24.sh $src_file $hypo_file $tgt_file $output_prefix score_xxl_24.log $cuda_devices
