
model_file=/raid_elmo/home/lr/lyu/checkpoints/meta19/en-ru/wmt19.en-ru.ensemble/model4.pt
data_path=/raid_elmo/home/lr/lyu/checkpoints/meta19/en-ru/wmt19.en-ru.ensemble/
mkdir /raid_elmo/home/lr/lyu/meta19_ct2
ct2-fairseq-converter --model_path $model_file --output_dir /raid_elmo/home/lr/lyu/meta19_ct2/en-ru --data_dir $data_path


model_file=//raid_elmo/home/lr/lyu/checkpoints/meta19/en-de/wmt19.en-de.joined-dict.ensemble/model4.pt
data_path=/raid_elmo/home/lr/lyu/checkpoints/meta19/en-de/wmt19.en-de.joined-dict.ensemble/
mkdir /raid_elmo/home/lr/lyu/meta19_ct2
ct2-fairseq-converter --model_path $model_file --output_dir /raid_elmo/home/lr/lyu/meta19_ct2/en-de --data_dir $data_path

model_file=//raid_elmo/home/lr/lyu/checkpoints/meta19/de-en/wmt19.de-en.joined-dict.ensemble/model4.pt
data_path=/raid_elmo/home/lr/lyu/checkpoints/meta19/de-en/wmt19.de-en.joined-dict.ensemble/
mkdir /raid_elmo/home/lr/lyu/meta19_ct2
ct2-fairseq-converter --model_path $model_file --output_dir /raid_elmo/home/lr/lyu/meta19_ct2/de-en --data_dir $data_path