from openai import OpenAI
from tqdm import tqdm
import json


OPENAI_API_KEY=""


client = OpenAI(api_key=OPENAI_API_KEY)
text_file_path="/raid/lyu/QEBT/en2de/wmt2020.en"
output_file_path="/local/lyu/wmt2020.en2de.gpt3.5turbo1106"
few_shot_demo_file_path="/home/lr/lyu/QEBT/paraphrasing/demonstration.txt"


with open(text_file_path, "r") as f:
    texts=[line.strip() for line in f.readlines()]
texts=texts[:5]
print("read {} lines".format(len(texts)))
with open(few_shot_demo_file_path, "r", encoding="utf-8") as f:
    few_shot_demo=f.read()
print("len of few shot demo: {}".format(len(few_shot_demo)))

prompt=few_shot_demo+"\n\nOriginal: "
results=[]
batch_size=5
for i in tqdm(range(0,len(texts),batch_size),desc="generating paraphrases",total=len(texts)//batch_size):
    batch=texts[i:i+batch_size]
    batch_input=[prompt+line+"\n" for line in batch]
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": batch_input},
    ]
    )
    for j in range(len(batch)):
        result={}
        result["original"]=batch[j]
        result["paraphrase"]=completion.choices[j].message.content
        results.append(result)
print(results)
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in results:
        f.write(json.dumps(line, ensure_ascii=False, indent=4))
        f.write("\n")
