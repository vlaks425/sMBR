# Train the model gpu23
set -e
cur_path=$(cd "$(dirname "$0")"; pwd)
input_prefix=/raid/lyu/fairseq-ru/baseline
output_prefix=/raid/lyu/fairseq-ru/baseline
checkpoints_dir=/local/lyu/checkpoints/en-ru/nc18/baseline
if [ ! -d $checkpoints_dir ]; then
    mkdir -p $checkpoints_dir
fi
CUDA_VISIBLE_DEVICES=0,1 fairseq-train $input_prefix/data-bin \
    --arch transformer --share-decoder-input-output-embed \
    --task translation \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12500 \
    --eval-bleu \
    --max-epoch 100 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --log-file $output_prefix/log \
    --sentencepiece-model-file $input_prefix/spm_joint.model \
    --update-freq 4 \
    --num-workers 4 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 1 \
    --ddp-backend no_c10d \
    --memory-efficient-fp16 \
    --source-lang en --target-lang ru \
    --eval-bleu-print-samples \
    --save-dir $checkpoints_dir \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

mv $checkpoints_dir/checkpoint_best.pt $input_prefix
bash $cur_path/evaluate.sh
rm -rf $checkpoints_dir