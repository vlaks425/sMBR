import deepl
from tqdm import tqdm
import json
source_language = 'EN'
target_language = 'DE'
pivot_as_src=['DE','FR','ES','IT','NL','PL','PT','RU','ZH','JA']
pivot_as_tgt=['DE','FR','ES','IT','NL','PL','PT-PT','RU','ZH','JA']
auth_key="a2e336e1-d5a5-9492-35a1-56118556cf6e:fx"
translator = deepl.Translator(auth_key)
text_file="/raid/lyu/QEBT/en2de/wmt2020.en"
output_file="/home/lr/lyu/QEBT/paraphrasing/en-de-test-deepl.txt"
with open(text_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines=[line.strip() for line in lines]
results=[]
for text in taqdm(lines,desc="translate",total=len(lines)):
    tmp_dict={}
    tmp_dict["text"]=text
    for pivot_lang_index in range(len(pivot_as_tgt)):
        result = translator.translate_text(text, target_lang=pivot_as_tgt[pivot_lang_index],source_lang=source_language)
        result = translator.translate_text(result.text, target_lang="EN-US",source_lang=pivot_as_src[pivot_lang_index])
        translated_text = result.text
        tmp_dict[pivot_as_tgt[pivot_lang_index]]=translated_text
    results.append(tmp_dict)
        

with open(output_file, 'w', encoding='utf-8') as f:
    for line in tqdm(results):
        f.write(json.dumps(line) + "\n")