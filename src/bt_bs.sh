#back translation with beam search
set -e
cur_path=$(cd "$(dirname "$0")"; pwd)
output_prefix=/raid/lyu/QEBT/bt_data/en-zh
input_prefix=$cur_path
checkpoints_dir=/raid_elmo/home/lr/lyu/checkpoints/en-zh/zh2en/baseline
spm_path=/raid/lyu/QEBT/en2zh/baseline/spm_joint.model
data_prefix=/raid/lyu/QEBT/zh2en/baseline
CUDA_VISIBLE_DEVICES=3 fairseq-interactive $data_prefix/data-bin \
    --input /raid/lyu/QEBT/news_crawl_zh/news.2022.zh.shuffled.deduped.300000.tok \
    --path $checkpoints_dir/checkpoint_best.pt \
    --num-workers 4 \
    --batch-size 100 --beam 5 --buffer-size 1800 > $output_prefix/news_crawl_zh_300k.bt.output.tmp
#extract the hypothesis
grep ^H $output_prefix/news_crawl_zh_300k.bt.output.tmp | cut -f3- > $output_prefix/news_crawl_zh_300k.bt.output
#detokenize the hypothesis
spm_decode --model=$spm_path --input_format=piece < $output_prefix/news_crawl_zh_300k.bt.output > $output_prefix/news_crawl_zh_300k.bt.output.detok