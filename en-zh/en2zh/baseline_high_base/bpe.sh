set -e
input_prefix=/raid/lyu/QEBT/en2zh/baseline_high
output_prefix=/raid/lyu/QEBT/en2zh/baseline_high
#cat $input_prefix/train.en $input_prefix/train.zh > $output_prefix/train.tokenized.en-zh
#spm_train --input=$input_prefix/train.tokenized.en-zh --model_prefix=$input_prefix/spm_joint --vocab_size=32000 --character_coverage=0.99995 --model_type=bpe --num_threads=28
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < $input_prefix/train.en > $input_prefix/train.bpe.en
spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < $input_prefix/train.zh > $input_prefix/train.bpe.zh
#spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-zh/test/en/test1.eng > $input_prefix/dev.bpe.en
#spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-zh/test/zh/test1.zho > $input_prefix/dev.bpe.zh
#spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-zh/test/en/test.en > $input_prefix/test.bpe.en
#spm_encode --model=$input_prefix/spm_joint.model --output_format=piece < /raid/lyu/en-zh/test/zh/test.zho > $input_prefix/test.bpe.zh
cut -f1 $input_prefix/spm_joint.vocab | tail -n +4 | sed "s/$/ 100/g" > $input_prefix/fairseq.vocab
# Binarize the dataset
fairseq-preprocess \
    --source-lang en --target-lang zh \
    --trainpref $input_prefix/train.bpe --validpref $input_prefix/dev.bpe --testpref $input_prefix/test.bpe \
    --destdir $input_prefix/data-bin --thresholdtgt 0 --thresholdsrc 0 \
    --workers 28 \
    --srcdict $input_prefix/fairseq.vocab \
    --joined-dictionary
rm $output_prefix/train.tokenized.en-zh
rm $input_prefix/train.bpe.en
rm $input_prefix/train.bpe.zh