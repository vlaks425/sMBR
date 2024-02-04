import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import ctranslate2
import sentencepiece as spm
from tqdm import tqdm
src_file="/raid/lyu/en-zh/test/en/test.en"
output_file="/home/lr/lyu/QEBT/debug/test.bt"
model_path = "/raid_elmo/home/lr/lyu/checkpoints/en-zh/en2zh/baseline/model_ct2.pt"
spm_model_path="/raid/lyu/QEBT/en2zh/baseline/spm_joint.model"
spm_model = spm.SentencePieceProcessor(model_file=spm_model_path)
translator = ctranslate2.Translator(model_path, device="cuda",device_index=[0,1] ,inter_threads=2,compute_type="float32")
with open(src_file, "r", encoding="utf-8") as fsrc:
    lines=fsrc.readlines()
    src=[]
    for line in tqdm(lines,desc="tokenize",total=len(lines)):
        line_tokenized = spm_model.encode(line, out_type=str)
        src.append(line_tokenized)
    src=src[:10]
print("successfully load src file")
results = translator.translate_batch(src,batch_type="tokens", max_batch_size=12000,beam_size=1,disable_unk=True,return_scores=True,sampling_topk=32001,num_hypotheses=160)
print("successfully translate")
results_detokenized=[]
results_score=[]
for result in tqdm(results,desc="detokenize",total=len(results)):
    for hypo in result:
        line=hypo["tokens"]
        score=hypo["score"]
        #把对数概率转换为概率
        score=2**score
        line = spm_model.decode(line)
        results_detokenized.append(" ".join(line.split()))
        results_score.append(score)
with open(output_file, "w", encoding="utf-8") as fout:
    for line in tqdm(results_detokenized):
        fout.write(line + "\n")
with open(output_file+".score", "w", encoding="utf-8") as fout:
    for line in tqdm(results_score):
        fout.write(str(line) + "\n")