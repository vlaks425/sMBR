import argparse
parser = argparse.ArgumentParser(description='perform sMBR')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--hypo_file', type=str, required=True, help='Path to the hypothese file')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')
parser.add_argument('--k', type=int, default=5, help='number of hypotheses per source sentence')
parser.add_argument('--qe_model', type=str, default="Unbabel/wmt22-cometkiwi-da", help='qe model')
parser.add_argument('--pp_model', type=str, default="lyu-boxuan/T5-sMBR-PP-EN", help='pp model')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--pp_batch_size', type=int, default=4, help='pp batch size')
parser.add_argument('--pp_num', type=int, default=16, help='pp num')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from comet import download_model, load_from_checkpoint
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
k=args.k
print("start to load src and hypo file")
with open(args.src_file, "r", encoding="utf-8") as fsrc:
    src_lines= fsrc.readlines()
    src_lines = [line.strip() for line in src_lines]
with open(args.hypo_file, "r", encoding="utf-8") as fhypo:
    hypo_lines=[line.strip() for line in fhypo.readlines()]
assert len(src_lines) == len(hypo_lines)/k, "len(src_lines): {}, len(hypo_lines): {}, k: {}".format(len(src_lines), len(hypo_lines), k)
tokenizer = AutoTokenizer.from_pretrained(args.pp_model)
pp_model = AutoModelForSeq2SeqLM.from_pretrained(args.pp_model,torch_dtype=torch.float16).to('cuda')
pp_model.eval()
pp_results = {}
for i in tqdm(range(0, len(src_lines), args.pp_batch_size), desc="Generating paraphrases"):
    batch = src_lines[i:i+args.pp_batch_size]
    encoding = tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt",truncation=True)
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    outputs = pp_model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_new_tokens=512,
        do_sample=True,
        epsilon_cutoff = 0.02,
        num_beams=1,
        num_return_sequences=args.pp_num
    )
    for _iter in range(len(batch)):
        original_sentence= batch[_iter]
        paraphrase_list=[]
        for output in outputs[_iter*args.pp_num:(_iter+1)*args.pp_num]:
            paraphrase = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            paraphrase_list.append(paraphrase)
        pp_results[original_sentence] = paraphrase_list
    del input_ids, attention_masks, outputs
    torch.cuda.empty_cache()
del pp_model
del tokenizer

data_list=[]
for i in range(len(src_lines)):
    original_src = src_lines[i]
    quasi_src = pp_results[original_src]
    srcs = [original_src] + quasi_src
    hypos = hypo_lines[i*k:(i+1)*k]
    for hypo in hypos:
        for src in srcs:
            data_list.append({"src": src, "mt": hypo})
    
    
assert len(data_list)==(len(src_lines) * k * (1 + args.pp_num)), "len(data_list): {}, len(src_lines): {}, k: {}, pp_num: {}".format(len(data_list), len(src_lines), k, args.pp_num)
print("successfully load data list")

model_path=download_model(args.qe_model)
model = load_from_checkpoint(model_path)
model.half()
model=torch.compile(model)
results=[]
number_gpu=len(args.gpu.split(","))
model_output = model.predict(
    data_list,
    batch_size=args.batch_size,
    gpus=number_gpu,
    num_workers=8,
)
for result in model_output.scores:
    results.append(result)
smbr_results=[]
for _iter in tqdm(range(len(hypo_lines)), desc="Calculating sMBR scores"):
    scores=results[_iter*(args.pp_num+1):(_iter+1)*(args.pp_num+1)]
    smbr_results.append(sum(scores) / len(scores))
assert len(results)==len(data_list), "len(results): {}, len(data_list): {}".format(len(results), len(data_list))
assert len(smbr_results) == len(hypo_lines), "len(smbr_results): {}, len(hypo_lines): {}".format(len(smbr_results), len(hypo_lines))
with open(args.output_file+".all_score", "w", encoding="utf-8") as fout:
    for score in smbr_results:
        fout.write(str(score) + "\n")

best_hypos = []
best_hypos_scores = []
for i in range(len(src_lines)):
    start = i * k
    end = (i + 1) * k
    scores = smbr_results[start:end]
    assert len(scores) == k, "len(scores): {}, k: {}".format(len(scores), k)
    best_index = scores.index(max(scores))
    best_hypos.append(hypo_lines[start + best_index])
    best_hypos_scores.append(scores[best_index])

with open(args.output_file, "w", encoding="utf-8") as fout:
    for hypo in best_hypos:
        fout.write(hypo.replace("\n", "") + "\n")

with open(args.output_file + ".score", "w", encoding="utf-8") as fout:
    for score in best_hypos_scores:
        fout.write(str(score) + "\n")

