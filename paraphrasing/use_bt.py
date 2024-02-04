import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--spm_path', type=str, default=None,help='Path to the sentencepiece model')
parser.add_argument('--gpu', type=str, default="0", help='GPU number to use, e.g., "0"')
parser.add_argument('--beam_size', type=int, required=True, default=1)
parser.add_argument('--num_hypotheses', type=int, required=True, default=1)
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--sentence_batch', type=bool, default=False)
parser.add_argument('--use_fastbpe', type=bool, default=False)
parser.add_argument('--bpe_codes', type=str, default=None)
parser.add_argument('--bpe_vocab', type=str, default=None)
parser.add_argument('--src_lang', type=str, default=None)
parser.add_argument('--tgt_lang', type=str, default=None)
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.use_gpu:
    print("using cuda device{}".format(args.gpu))
gpu_num=len(args.gpu.split(","))
import ctranslate2
import sentencepiece as spm
import fastBPE
from sacremoses import MosesDetokenizer, MosesTokenizer
from tqdm import tqdm
import math
src_file=args.src_file
output_file=args.output_file
model_path = args.model_path
spm_model=None
if args.spm_path is not None:
    spm_model_path=args.spm_path
    spm_model = spm.SentencePieceProcessor(model_file=spm_model_path)
if args.use_fastbpe:
    bpe = fastBPE.fastBPE(args.bpe_codes, args.bpe_vocab)
    mt= MosesTokenizer(lang=args.src_lang)
    md = MosesDetokenizer(lang=args.tgt_lang)
    bpe_symbol = "@@ "
    

def fast_bpe_detokenize(line):
    return (line + " ").replace(bpe_symbol, "").rstrip()
    

def mosesdetokenize(line):
    return md.detokenize(line.split())


if args.use_gpu:
    if gpu_num==1:
        translator = ctranslate2.Translator(model_path, device="cuda",device_index=[0])
    elif gpu_num==2:
        translator = ctranslate2.Translator(model_path, device="cuda",device_index=[0,1])
else:
    translator = ctranslate2.Translator(model_path, device="cpu",intra_threads=min(28,os.cpu_count()))
    

with open(src_file, "r", encoding="utf-8") as fsrc:
    lines=fsrc.readlines()
    src=[]
    if spm_model is not None:
        for line in tqdm(lines,desc="tokenize",total=len(lines)):
            line= spm_model.encode(line.strip(), out_type=str)
            src.append(line)
    elif args.use_fastbpe:
        for line in tqdm(lines,desc="tokenize",total=len(lines)):
            line=mt.tokenize(line.strip(), return_str=True,aggressive_dash_splits=True,escape=True)
            line=bpe.apply([line])[0]
            src.append(line.split())
print("successfully load src file")
batch_size=32
results=[]
for i in tqdm(range(len(src)//batch_size+1),desc="translate",total=len(src)//batch_size+1):
    if args.use_gpu:
        if args.sentence_batch:
            results.extend(translator.translate_batch(src[i*batch_size:(i+1)*batch_size],batch_type="examples", max_batch_size=args.batch_size,beam_size=args.beam_size, num_hypotheses=args.num_hypotheses,disable_unk=True,return_scores=True))
        else:
            results.extend(translator.translate_batch(src[i*batch_size:(i+1)*batch_size],batch_type="tokens", max_batch_size=args.batch_size,beam_size=args.beam_size, num_hypotheses=args.num_hypotheses,disable_unk=True,return_scores=True))
    else:
        results.extend(translator.translate_batch(src[i*batch_size:(i+1)*batch_size],batch_type="tokens",  max_batch_size=args.batch_size,beam_size=args.beam_size, num_hypotheses=args.num_hypotheses,disable_unk=True,return_scores=True))
        

print("successfully translate")
results_detokenized=[]
results_scores=[]
for result in tqdm(results,desc="detokenize",total=len(results)):
    for hypo in result:
        line=hypo["tokens"]
        score=hypo["score"]
        score=math.exp(score)
        if spm_model is not None:
            line = spm_model.decode(line)
        else:
            line=" ".join(line)
        results_detokenized.append(line)
        results_scores.append(score)
#assert len(results_detokenized)==len(src)
with open(output_file, "w", encoding="utf-8") as fout:
    for line in tqdm(results_detokenized):
        fout.write(line + "\n")
with open(output_file+".score", "w", encoding="utf-8") as fout:
    for line in tqdm(results_scores):
        fout.write(str(line) + "\n")