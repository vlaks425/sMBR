from openai import OpenAI
from tqdm import tqdm
import json

OPENAI_API_KEY=

client = OpenAI(api_key=OPENAI_API_KEY)
#text_file_path="/raid/lyu/QEBT/en2de/wmt2020.en"
text_file_path="/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en"
output_file_path="/local/lyu/wmt2023.en2ru.gpt-4-0125-preview"
#output_file_path="/local/lyu/wmt2023.en2de.gpt-4-0125-preview"
few_shot_demo_file_path="/home/lr/lyu/QEBT/paraphrasing/demonstration.txt"
copy2file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview"
#copy2file_path="/raid/lyu/QEBT/en2de/wmt2020.en2de.gpt-4-0125-preview"

with open(text_file_path, "r") as f:
    texts=[line.strip() for line in f.readlines()]
print("read {} lines".format(len(texts)))
with open(few_shot_demo_file_path, "r", encoding="utf-8") as f:
    few_shot_demo=f.read()
print("len of few shot demo: {}".format(len(few_shot_demo)))

prompt=few_shot_demo+"\n\nOriginal: "
results=[]
batch_size=5
for line in tqdm(texts,desc="generating paraphrases",total=len(texts)):
    result={}
    completion = client.chat.completions.create(
    model="gpt-4-0125-preview",
    seed=114514,
    max_tokens=2048,
    messages=[
        {"role": "user", "content": prompt+line+"\n"},
    ]
    )
    result["original"]=line
    result["paraphrase"]=completion.choices[0].message.content
    results.append(result)
#print(results)
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in results:
        f.write(json.dumps(line, ensure_ascii=False))
        f.write("\n")
        
#copy file to copy2file_path
import shutil
shutil.copy(output_file_path,copy2file_path)
