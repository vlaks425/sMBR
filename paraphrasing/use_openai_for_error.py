from openai import OpenAI
from tqdm import tqdm
import json

def is_output_valid(output):
    output=output.split("\n")
    try:
        rephrases = [line.split(":", 1)[1].strip() for line in output if "Rephrase" in line]
    except:
        return False
    if len(rephrases)!=5:
        return False
    return True


OPENAI_API_KEY=""

client = OpenAI(api_key=OPENAI_API_KEY)
text_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview.json.error"
output_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview.error.fixed.json"
few_shot_demo_file_path="/home/lr/lyu/QEBT/paraphrasing/demonstration.txt"


with open(text_file_path, "r") as f:
    texts=[line.strip() for line in f.readlines()]
    texts=[json.loads(line) for line in texts]
    texts=[line["original"] for line in texts]
print("read {} lines".format(len(texts)))
with open(few_shot_demo_file_path, "r", encoding="utf-8") as f:
    few_shot_demo=f.read()
print("len of few shot demo: {}".format(len(few_shot_demo)))

prompt=few_shot_demo+"\n\nOriginal: "
results=[]
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
final_results=[]
for i in range(len(results)):
    seed=114514
    generated_output=results[i]["paraphrase"]
    if is_output_valid(generated_output):
        final_results.append(results[i])
        continue
    while True:
        seed+=1
        print("retrying")
        line=results[i]["original"]
        print("retrying")
        completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        max_tokens=2048,
        seed=seed,
        messages=[
            {"role": "user", "content": prompt+line+"\n"},
        ]
        )
        generated_output=completion.choices[0].message.content
        if is_output_valid(generated_output):
            print("retry success")
            final_results.append({"original":line,"paraphrase":generated_output})
            break
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in final_results:
        f.write(json.dumps(line, ensure_ascii=False))
        f.write("\n")
