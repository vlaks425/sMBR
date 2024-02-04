import json
from tqdm import tqdm
input_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview"
output_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview.json"
inputs=[]
with open(input_file_path,"r",encoding="utf-8") as f:
    lines=f.readlines()
    for line in lines:
        inputs.append(json.loads(line.strip()))
print("read {} lines".format(len(inputs)))
results=[]
error=[]
for input_line in tqdm(inputs,desc="postprocessing"):
    paraphrase=input_line["paraphrase"]
    original=input_line["original"]
    paraphrase=paraphrase.split("\n")
    paraphrase=[line.strip() for line in paraphrase if len(line.strip())>0]
    #paraphrase=paraphrase[:5]
    try:
        rephrases = [line.split(":", 1)[1].strip() for line in paraphrase if "Rephrase" in line]
    except:
        error.append({"original":original,"paraphrase":rephrases})
        continue
    if len(rephrases)!=5:
        error.append({"original":original,"paraphrase":rephrases})
        continue
    results.append({"original":original,"paraphrase":rephrases})
print("error {} lines".format(len(error)))
print("write {} lines".format(len(results)))
with open(output_file_path,"w",encoding="utf-8") as f:
    f.write(json.dumps(results,ensure_ascii=False,indent=4))
if len(error)==0:
    exit()
with open(output_file_path+".error","w",encoding="utf-8") as f:
    for line in error:
        f.write(json.dumps(line,ensure_ascii=False))
        f.write("\n")
    