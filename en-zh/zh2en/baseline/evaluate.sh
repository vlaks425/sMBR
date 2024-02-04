set -e
cur_path=$(cd "$(dirname "$0")"; pwd)
data_prefix=/raid/lyu/QEBT/zh2en/baseline
ref_file=/raid/lyu/en-zh/test/en/test.en
src_file=/raid/lyu/en-zh/test/zh/test.zho
input_prefix=$cur_path
checkpoints_dir=/raid_elmo/home/lr/lyu/checkpoints/en-zh/zh2en/baseline

#CUDA_VISIBLE_DEVICES=3 fairseq-interactive $data_prefix/data-bin \
    #--input $data_prefix/test.bpe.zh \
    #--path $checkpoints_dir/checkpoint_best.pt \
    #--num-workers 4 \
    #--batch-size 128 --beam 5 --buffer-size 512 > $input_prefix/test.output
#grep ^H $input_prefix/test.output | cut -f3- > $input_prefix/test.output1
#spm_decode --model=$data_prefix/spm_joint.model --input_format=piece < $input_prefix/test.output1 > $input_prefix/output
#sacrebleu $ref_file -i $input_prefix/output -m bleu chrf ter > $input_prefix/score.log
#CUDA_VISIBLE_DEVICES=3 comet-score -s $src_file -t $input_prefix/output -r $ref_file  --quiet --only_system >> $input_prefix/score.log
CUDA_VISIBLE_DEVICES=3 python /raid/lyu/evaluate_entity_translation/evaluate_entity_translation.py --ref_file $ref_file --hypo_file $input_prefix/output --lang en >> $input_prefix/score.log
CUDA_VISIBLE_DEVICES=3 bert-score -r $ref_file -c $input_prefix/output --idf --lang en >> $input_prefix/score.log

