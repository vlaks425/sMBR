set -e

src_file=/raid/lyu/QEBT/en2de/wmt2020.en
tgt_file=/raid/lyu/QEBT/en2de/wmt2020.de
decoding_method=bs
cuda_devices=0
output_prefix=/raid/lyu/QEBT/test_dev/test_en2de
baseline_400=/raid/lyu/QEBT/test_dev/test_en2de/bs/400
qe_400_output_prefix=/raid/lyu/QEBT/test_dev/test_en2de/qe_wmt23-cometkiwi-da-xl_$decoding_method/output_400
# 定义一个包含不同 k 值的数组
k_values=(5 16 32 64 80 96 128 160 200 240 280 320 340 360 400)
#python /home/lr/lyu/QEBT/src/use_hf_model.py --src_file $src_file --output_file $output_prefix --gpu $cuda_devices --k 400
mkdir -p $output_prefix/qe_wmt23-cometkiwi-da-xl_$decoding_method/output_400
#python /home/lr/lyu/QEBT/src/QE_comet.py --src_file $src_file --hypo_file $baseline_400/hypo_400 --output_file $qe_400_output_prefix/qe.output --gpu $cuda_devices --k 400
for k in "${k_values[@]}"
do
    qe_output_prefix=$output_prefix/qe_wmt23-cometkiwi-da-xl_$decoding_method/output_$k
    baseline_output_prefix=$output_prefix/$decoding_method/$k
    mkdir -p $baseline_output_prefix
    mkdir -p $qe_output_prefix
    if [ $k -eq 400 ]; then
        echo "k=400, so copy hypo_400 to $qe_output_prefix"
        #cp $baseline_output_prefix/hypo_$k $qe_output_prefix
    else
        awk "NR%400>=1 && NR%400<=$k" $baseline_400/hypo_400 > $qe_output_prefix/hypo_$k
        awk "NR%400>=1 && NR%400<=$k" $qe_400_output_prefix/qe.output.score > $qe_output_prefix/qe.output.score
        python /home/lr/lyu/QEBT/src/extract_hypo_with_score.py --input_file $qe_400_output_prefix/hypo_400 --output_file $qe_output_prefix --score_file $qe_400_output_prefix/qe.output.score --num_hypotheses 400 --k $k 
    fi
    awk "NR%400==1" $baseline_400/hypo_400 > $baseline_output_prefix/hypo_1

    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $qe_output_prefix/qe.output $qe_output_prefix $cuda_devices
    bash /home/lr/lyu/QEBT/src/eval_de.sh $src_file $tgt_file $baseline_output_prefix/hypo_1 $baseline_output_prefix $cuda_devices
done
    
