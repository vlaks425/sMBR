import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')
parser.add_argument('--k', type=int, default=5, help='k-best')
parser.add_argument('--decoding_method', type=str, default="bs", help='decoding method')
parser.add_argument('--num_hypotheses', type=int, required=True, default=1)
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from typing import  Optional, Tuple
from transformers import set_seed
set_seed(114514)
from tqdm import tqdm
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-en-de"
decoding_method = args.decoding_method
k=args.k
num_hypotheses=args.num_hypotheses
src_file = args.src_file
output_path = args.output_file
#如果不存在output_path，则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_path + decoding_method):
    os.makedirs(output_path + decoding_method)
if not os.path.exists(output_path + decoding_method+"/"+str(k)):
    os.makedirs(output_path + decoding_method+"/"+str(k))
output_file = output_path + decoding_method+"/"+str(k)+"/hypo_"+str(k)

batch_size = 1
print("Using model: %s" % mname)
tokenizer = FSMTTokenizer.from_pretrained(mname,cache_dir="/cache01/lyu/checkpoints")
model = FSMTForConditionalGeneration.from_pretrained(mname,cache_dir="/cache01/lyu/checkpoints").to("cuda")
print("Model loaded.")


with open(src_file, "r", encoding="utf-8") as f:
    results = []
    scores=[]
    src_lines = f.readlines()
    src_lines = [line.strip() for line in src_lines]
    for i in tqdm(range(len(src_lines)//batch_size+1),total=len(src_lines)//batch_size+1,desc="预测"):
        batch = src_lines[i*batch_size:(i+1)*batch_size]
        if len(batch) == 0:
            continue
        with torch.no_grad():
            batch= tokenizer(batch, return_tensors="pt", padding=True,truncation=True).to("cuda")
            batch_results = model.generate(**batch,max_length=200,num_beams=k,output_scores=True,return_dict_in_generate=True,num_return_sequences=num_hypotheses,do_sample=False)
            result = tokenizer.batch_decode(batch_results.sequences, skip_special_tokens=True)
            batch_scores=batch_results.sequences_scores.cpu().numpy()
            batch_scores=np.exp(batch_scores).tolist()
            scores.extend(batch_scores)
            batch_results = tokenizer.batch_decode(batch_results.sequences, skip_special_tokens=True)
            results.extend(batch_results)
            assert len(results) == len(scores)
        del batch
        del batch_results
        torch.cuda.empty_cache()
    #assert len(results) == len(src_lines),f"len(results)={len(results)},len(src_lines)={len(src_lines)}"
with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line+"\n")
with open(output_file+".score", "w", encoding="utf-8") as f:
    for line in scores:
        f.write(str(line)+"\n")