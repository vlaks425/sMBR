import argparse
parser = argparse.ArgumentParser(description='perform sMBR')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--hypo_file', type=str, required=True, help='Path to the hypothese file')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')
parser.add_argument('--src_num', type=int, default=5, help='src num')
parser.add_argument('--qe_model', type=str, default="Unbabel/wmt22-cometkiwi-da", help='qe model')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from comet import download_model, load_from_checkpoint
import torch
from tqdm import tqdm
import json
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.benchmark = True
k_list=[400,340,320, 280, 240, 200, 160, 128, 96, 80, 64, 32, 16, 5]
src_num=args.src_num
hypo_file=args.hypo_file
src_file=args.src_file
output_file=args.output_file
model_path=download_model(args.qe_model,saving_directory="/cache01/lyu/comet_model")
# model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl",saving_directory="/cache01/lyu/comet_model")
#model_path=download_model("Unbabel/wmt22-cometkiwi-da",saving_directory="/cache01/lyu/comet_model")
#model_path=download_model("Unbabel/wmt20-comet-qe-da",saving_directory="/cache01/lyu/comet_model")
model = load_from_checkpoint(model_path)
model=torch.compile(model)
print("start to load src and hypo file")
with open(src_file, "r", encoding="utf-8") as fsrc:
    src_dict=json.load(fsrc)
    src=[]
    for sample in src_dict:
        src.append(sample["paraphrase"])
for k in k_list:
with open(hypo_file, "r", encoding="utf-8") as fhypo:
    hypo=[line.strip() for line in fhypo.readlines()]
assert (k*len(src))==len(hypo)
group_num=len(src)
assert group_num==(len(hypo)//(k))
print("successfully load src and hypo file")
data_list=[]
for i in tqdm(range(group_num)):
    for j in range(k):
        for src_text in src[i]:
            data_list.append({"src":src_text,"mt":hypo[i*k+j]})
assert len(data_list)==(group_num*src_num*k)
print("successfully load data list")


results=[]
number_gpu=len(args.gpu.split(","))
model_output = model.predict(data_list, batch_size=args.batch_size,gpus=number_gpu)
for result in model_output.scores:
    results.append(result)
smbr_results=[]
for _iter in tqdm(range(len(hypo))):
    #这是某一个假设与对应的src_num个源句的分数
    scores=results[_iter*src_num:_iter*src_num+src_num]
    smbr_results.append(scores)
assert len(results)==len(data_list)
with open(output_file+".score", "w", encoding="utf-8") as fout:
    for result in tqdm(smbr_results):
        fout.write(json.dumps(result) + "\n")