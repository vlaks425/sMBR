import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--hypo_file', type=str, required=True, help='Path to the hypothese file')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')
parser.add_argument('--k', type=int, default=5, help='k-best')
parser.add_argument('--qe_model', type=str, default="Unbabel/wmt22-cometkiwi-da", help='qe model')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from comet import download_model, load_from_checkpoint
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
k=args.k
hypo_file=args.hypo_file
src_file=args.src_file
output_file=args.output_file

print("start to load src and hypo file")

with open(src_file, "r", encoding="utf-8") as fsrc:
    src=[line.strip() for line in fsrc.readlines()]

with open(hypo_file, "r", encoding="utf-8") as fhypo:
    hypo=[line.strip() for line in fhypo.readlines()]
assert (k*len(src))==len(hypo),f"src:{len(src)},hypo:{len(hypo)},k:{k}"

print("successfully load src and hypo file")
data_list=[]
for i in tqdm(range(len(src))):
    for j in range(k):
        data_list.append({"src":src[i],"mt":hypo[i*k+j]})
print("successfully load data list")

model_path=download_model(args.qe_model)
model = load_from_checkpoint(model_path)
model.half()
model=torch.compile(model)
results=[]
number_gpu=len(args.gpu.split(","))
model_output = model.predict(data_list, batch_size=args.batch_size,gpus=number_gpu,num_workers=8)

for result in model_output.scores:
    results.append(result)
assert len(results)==len(data_list)

with open(output_file+".score", "w", encoding="utf-8") as fout:
    for result in tqdm(results):
        fout.write(str(result) + "\n")

with open(output_file, "w", encoding="utf-8") as fout:
    if k!=1:
        for i in tqdm(range(len(results)//k)):
            candidate_scores = results[i*k:i*k+k]
            best_index = candidate_scores.index(max(candidate_scores))
            fout.write(data_list[i*k+best_index]["mt"] + "\n")
