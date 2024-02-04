import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import ctranslate2
import sentencepiece as spm
from tqdm import tqdm
im
src_file="/raid/lyu/QEBT/news_crawl_en/news.2022.en.shuffled.deduped.300000.tok"
output_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_en_300k_hypo_5.bt"
model_path = "/raid_elmo/home/lr/lyu/checkpoints/en-zh/en2zh/baseline/model_ct2.pt"
spm_model_path="/raid/lyu/QEBT/en2zh/baseline/spm_joint.model"
spm_model = spm.SentencePieceProcessor(model_file=spm_model_path)
translator = ctranslate2.Translator(model_path, device="cuda", device_index=[0],compute_type="float16")
with open(src_file, "r", encoding="utf-8") as fsrc:
    src=[line.strip().split() for line in fsrc.readlines()]
print("successfully load src file")
batch_size=10000
results=[]
for i in tqdm(range(len(src)//batch_size+1),desc="translate",total=len(src)//batch_size+1):
    results.extend(translator.translate_batch(src[i*batch_size:(i+1)*batch_size],batch_type="tokens", max_batch_size=12000,beam_size=5, num_hypotheses=5,disable_unk=True,return_scores=True))

print("successfully translate")
results_detokenized=[]
results_scores=[]
for result in tqdm(results,desc="detokenize",total=len(results)):
    for hypo in result:
        line=hypo["tokens"]
        score=hypo["score"]
        score=2**score
        line = spm_model.decode(line)
        results_detokenized.append(line)
        results_scores.append(score)
#assert len(results_detokenized)==len(src)
with open(output_file, "w", encoding="utf-8") as fout:
    for line in tqdm(results_detokenized):
        fout.write(line + "\n")
with open(output_file+".score", "w", encoding="utf-8") as fout:
    for line in tqdm(results_scores):
        fout.write(str(line) + "\n")