set -e
input_prefix=/raid/lyu/fairseq-ru/baseline
checkpoints_dir=/raid/lyu/fairseq-ru/baseline
CUDA_VISIBLE_DEVICES=0 fairseq-interactive $input_prefix/data-bin \
    --input $input_prefix/test.bpe.en \
    --path $checkpoints_dir/checkpoint_best.pt \
    --num-workers 4 \
    --batch-size 128 --beam 5 --buffer-size 512 > $input_prefix/test.output
grep ^H $input_prefix/test.output | cut -f3- > $input_prefix/test.output1
spm_decode --model=$input_prefix/spm_joint.model --input_format=piece < $input_prefix/test.output1 > $input_prefix/output.ru
sacrebleu /raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru -i $input_prefix/output.ru -m bleu chrf ter > $input_prefix/score.log
CUDA_VISIBLE_DEVICES=0 comet-score -s /raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en -t $input_prefix/output.ru -r /raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru --quiet --only_system >> $input_prefix/score.log
CUDA_VISIBLE_DEVICES=0 python /raid/lyu/evaluate_entity_translation/evaluate_entity_translation.py --ref_file /raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru --hypo_file $input_prefix/output.ru --lang ru >> $input_prefix/score.log
CUDA_VISIBLE_DEVICES=0 bert-score -r /raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru -c $input_prefix/output.ru --idf --lang ru >> $input_prefix/score.log
