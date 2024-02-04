set -e
cuda_device=0
cur_path=$(cd "$(dirname "$0")"; pwd)
data_prefix=/raid/lyu/QEBT/en2zh/bt_news_crawl_300k_MBR_hypo_5
ref_file=/raid/lyu/en-zh/test/zh/test.zho
input_prefix=$cur_path
checkpoints_dir=/raid_elmo/home/lr/lyu/checkpoints/en-zh/en2zh/bt_news_crawl_300k_MBR_hypo_5
CUDA_VISIBLE_DEVICES=$cuda_device fairseq-interactive $data_prefix/data-bin \
    --input $data_prefix/test.bpe.en \
    --path $checkpoints_dir/checkpoint_best.pt \
    --num-workers 4 \
    --batch-size 128 --beam 5 --buffer-size 512 > $input_prefix/test.output
grep ^H $input_prefix/test.output | cut -f3- > $input_prefix/test.output1
spm_decode --model=$data_prefix/spm_joint.model --input_format=piece < $input_prefix/test.output1 > $input_prefix/output.detok
python /raid/lyu/fairseq-zh/test.zh.py --input "$input_prefix/output.detok" --output "$input_prefix/output.zh"
sacrebleu /raid/lyu/en-zh/test/zh/test.zho -i $input_prefix/output.zh -m bleu chrf ter -tok zh > $input_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_device comet-score -s /raid/lyu/en-zh/test/en/test.en -t $input_prefix/output.zh -r $ref_file  --quiet --only_system >> $input_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_device python /raid/lyu/evaluate_entity_translation/evaluate_entity_translation.py --ref_file $ref_file --hypo_file $input_prefix/output.zh --lang zh >> $input_prefix/score.log
CUDA_VISIBLE_DEVICES=$cuda_device bert-score -r $ref_file -c $input_prefix/output.zh --idf --lang zh >> $input_prefix/score.log

