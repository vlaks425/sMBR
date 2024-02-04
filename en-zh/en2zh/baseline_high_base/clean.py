#清理平行语料库
import re
from tqdm import tqdm

output_path="/raid/lyu/QEBT/en2zh/baseline_high_big"


un_en="/raid/lyu/en-zh/UN/en-zh/UNv1.0.en-zh.en"
un_zh="/raid/lyu/en-zh/UN/en-zh/UNv1.0.en-zh.zh"
wikititles_en="/raid/lyu/en-zh/wikititle/wikititles-v3.en"
wikititles_zh="/raid/lyu/en-zh/wikititle/wikititles-v3.zh"
nc18_en="/raid/lyu/en-zh/nc18/train.en.clean"
nc18_zh="/raid/lyu/en-zh/nc18/train.zh.clean"
ccmatrix_en="/raid/lyu/en-zh/ccmatrix/cleaned.en"
ccmatrix_zh="/raid/lyu/en-zh/ccmatrix/cleaned.zh"

file_list_en=[un_en,wikititles_en,nc18_en,ccmatrix_en]
file_list_zh=[un_zh,wikititles_zh,nc18_zh,ccmatrix_zh]
en_result=[]
zh_result=[]
for en,zh in tqdm(zip(file_list_en,file_list_zh)):
    with open(en, "r", encoding="utf-8") as f, open(zh, "r", encoding="utf-8") as f2:
        lines=[line.strip() for line in f.readlines()]
        lines2=[line.strip() for line in f2.readlines()]
        for line,line2 in tqdm(zip(lines,lines2),total=len(lines)):
            if line and line2:
                en_len=len(line.split())
                zh_len=len(line2)
                if en_len>200 or zh_len>200:
                    continue
                en_result.append(line)
                zh_result.append(line2)
assert len(en_result)==len(zh_result)
result=set()
for line1,line2 in tqdm(zip(en_result,zh_result)):
    result.add((line1,line2))
with open(output_path+"/train.zh", "w", encoding="utf-8") as fout, open(output_path+"/train.en", "w", encoding="utf-8") as fout2:
    for line1,line2 in tqdm(result):
        fout.write(line2+"\n")
        fout2.write(line1+"\n")