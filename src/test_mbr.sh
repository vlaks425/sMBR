#test script for MBR decoding
set -e
#cat /raid/lyu/QEBT/news_crawl_zh/news.2022.zh.shuffled.deduped.300000 | head -100 > /home/lr/lyu/QEBT/debug/test_mbr.zh
#cat /raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt | head -500 > /home/lr/lyu/QEBT/debug/test_mbr.bt
comet-mbr \
    -s /home/lr/lyu/QEBT/debug/test_mbr.zh -t /home/lr/lyu/QEBT/debug/test_mbr.bt \
    --num_sample 5 -o /home/lr/lyu/QEBT/debug/test_mbr.output \
    --gpus 1  --qe_model Unbabel/wmt23-cometkiwi-da-xl \
    --model_storage_path /cache01/lyu/comet_model
