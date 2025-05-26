set -e
src_file=data/wmt2023/en-de/generaltest2023.en-de.src.en
tgt_file=data/wmt2023/en-de/generaltest2023.en-de.ref.refA.de
decoding_method=eps
cuda_devices=0
qe_model=Unbabel/wmt22-cometkiwi-da
pp_model=lyu-boxuan/T5-sMBR-PP-EN
k_values=(128 5)
pp_num=16
for k in "${k_values[@]}"
do
  echo "start $k"
  output_prefix=output/tower13b_en2de/smbr_pp_$pp_num_$decoding_method/$k
  baseline_prefix=output/tower13b_en2de/$decoding_method/$k
  mkdir -p $output_prefix
  python src/sMBR-PP.py --src_file $src_file --hypo_file $baseline_prefix/hypo_$k --output_file $output_prefix/smbr.output --gpu $cuda_devices --k $k --qe_model $qe_model --batch_size 256 --pp_model $pp_model --pp_num $pp_num
  bash src/eval.sh $src_file $tgt_file $output_prefix/smbr.output $output_prefix $cuda_devices
  echo "done $k"

done
