# Train the model gpu07
set -e
ulimit -n 10240
bash /home/lr/lyu/QEBT/en-zh/bt_news_crawl_300k_MBR_hypo_5/bpe.sh
cur_path=$(cd "$(dirname "$0")"; pwd)
input_prefix=/raid/lyu/QEBT/en2zh/bt_news_crawl_300k_MBR_hypo_5
output_prefix=$cur_path
checkpoints_dir=/cache01/lyu/checkpoints/en2zh/nc18/bt_news_crawl_300k_MBR_hypo_5
mv_dir=/raid_elmo/home/lr/lyu/checkpoints/en-zh/en2zh/bt_news_crawl_300k_MBR_hypo_5
mkdir -p $mv_dir
if [ ! -d $checkpoints_dir ]; then
    mkdir -p $checkpoints_dir
fi
CUDA_VISIBLE_DEVICES=0,1 fairseq-train $input_prefix/data-bin \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 25000 \
    --eval-bleu \
    --max-epoch 150 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --log-file $output_prefix/log \
    --sentencepiece-model-file $input_prefix/spm_joint.model \
    --update-freq 2 \
    --num-workers 4 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 1 \
    --ddp-backend no_c10d \
    --memory-efficient-fp16 \
    --source-lang en --target-lang zh \
    --eval-bleu-print-samples \
    --save-dir $checkpoints_dir \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

mv $checkpoints_dir/checkpoint_best.pt $mv_dir
bash $cur_path/evaluate.sh
rm -rf $checkpoints_dir