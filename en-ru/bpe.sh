set -e
input_prefix=/raid/lyu/fairseq-ru/baseline

cat /raid/lyu/en-ru/nc18/train.en.clean /raid/lyu/en-ru/nc18/train.ru.clean > $input_prefix/train.tokenized.ru-en
spm_train --input=$input_prefix/train.tokenized.ru-en --model_prefix=$input_prefix/spm_joint --vocab_size=32000 --character_coverage=1.0 --model_type=bpe --num_threads=16
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-ru/nc18/train.ru.clean > $input_prefix/train.bpe.ru
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-ru/nc18/train.en.clean > $input_prefix/train.bpe.en
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-ru/test.en > $input_prefix/dev.bpe.en
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-ru/test.ru > $input_prefix/dev.bpe.ru
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-ru/tests/generaltest2023.en-ru.ref.refA.ru > $input_prefix/test.bpe.ru
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en > $input_prefix/test.bpe.en
# Binarize the dataset
cut -f1 $input_prefix/spm_joint.vocab | tail -n +4 | sed "s/$/ 100/g" > $input_prefix/fairseq.vocab
fairseq-preprocess \
    --source-lang en --target-lang ru \
    --trainpref $input_prefix/train.bpe --validpref $input_prefix/dev.bpe --testpref $input_prefix/test.bpe \
    --destdir $input_prefix/data-bin --thresholdtgt 0 --thresholdsrc 0 \
    --workers 28 \
    --srcdict $input_prefix/fairseq.vocab \
    --joined-dictionary
rm -rf $input_prefix/train.tokenized.ru-en