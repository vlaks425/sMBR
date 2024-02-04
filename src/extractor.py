from tqdm import tqdm
import random
import sentencepiece as spm
spm_model_path="/raid/lyu/QEBT/en2zh/baseline/spm_joint.model"
input_file = '/raid/lyu/QEBT/news_crawl_en/news.2022.en.shuffled.deduped'
k=300000
random.seed(1234)
with open(input_file, 'r',encoding="utf-8") as fin:
    lines = [line.strip() for line in fin.readlines()]
    lines = [line for line in lines if len(line) > 0]
    total = len(lines)
    index=random.sample(range(total),k)
bach_translation=[]
for i in tqdm(index):
    bach_translation.append(lines[i])
s = spm.SentencePieceProcessor(model_file=spm_model_path)
bach_translation_tokenized=[]
for line in tqdm(bach_translation):
    line_tokenized = s.encode(line, out_type=str)
    line_tokenized = " ".join(line_tokenized)
    bach_translation_tokenized.append(line_tokenized)
result=[]
for line1,line2 in tqdm(zip(bach_translation,bach_translation_tokenized)):
    if len(line2.split())>1000:
        continue
    result.append((line1,line2))
with open(input_file+"."+str(k), 'w',encoding="utf-8") as fout1,open(input_file+"."+str(k)+".tok", 'w',encoding="utf-8") as fout2:
    for line1,line2 in tqdm(result):
        fout1.write(line1+"\n")
        fout2.write(line2+"\n")
