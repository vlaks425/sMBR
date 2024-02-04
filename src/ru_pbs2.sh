set -e


src_file=/raid/lyu/en-ru/tests/Statmt-newstest_enru-2020-eng-rus.eng
tgt_file=/raid/lyu/en-ru/tests/Statmt-newstest_enru-2020-eng-rus.rus

decoding_method=bs
cuda_devices=0,1


baseline_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/bs/5

group_baseline=/raid/lyu/QEBT/test_dev/tune_test_en2ru/$decoding_method/400
output_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/mbr_bs/output_80
bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix ru $cuda_devices
bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs ru $cuda_devices


decoding_method=as

group_baseline=/raid/lyu/QEBT/test_dev/tune_test_en2ru/$decoding_method/400
output_prefix=/raid/lyu/QEBT/test_dev/tune_test_en2ru/mbr_as/output_80
bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $baseline_prefix/hypo_1 $baseline_prefix $output_prefix ru $cuda_devices
bash /home/lr/lyu/QEBT/src/eval_pbs.sh $src_file $tgt_file $output_prefix/mbr.output $group_baseline/hypo_1 $group_baseline $output_prefix/group_pbs ru $cuda_devices

