import json
from tqdm import tqdm
input_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview"
error_fixed_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview.error.fixed.json"
output_file_path="/raid/lyu/QEBT/en2ru/wmt2023.en2ru.gpt-4-0125-preview.json"
inputs=[]
with open(input_file_path,"r",encoding="utf-8") as f:
    lines=f.readlines()
    for line in lines:
        inputs.append(json.loads(line.strip()))
    #for i in range(0,len(lines),4):
        #dict_=json.loads(lines[i].strip()+lines[i+1].strip()+lines[i+2].strip()+lines[i+3].strip())
        #inputs.append(dict_)
print("read {} lines".format(len(inputs)))
error_fix_lines={}
with open(error_fixed_file_path,"r",encoding="utf-8") as f:
    lines=f.readlines()
    for line in lines:
        line=json.loads(line.strip())
        error_fix_lines[line["original"]]=line["paraphrase"]
print("read {} lines".format(len(error_fix_lines)))
results=[]
for input_line in tqdm(inputs,desc="postprocessing"):
    paraphrase=input_line["paraphrase"]
    original=input_line["original"]
    paraphrase=paraphrase.split("\n")
    paraphrase=[line.strip() for line in paraphrase if len(line.strip())>0]
    try:
        rephrases = [line.split(":", 1)[1].strip() for line in paraphrase if "Rephrase" in line]
        rephrases=rephrases[:5]
        assert len(rephrases)==5
        results.append({"original":original,"paraphrase":rephrases})
    except:
        rephrases_line=error_fix_lines[original]
        rephrases_line=rephrases_line.split("\n")
        rephrases = [line.split(":", 1)[1].strip() for line in rephrases_line if "Rephrase" in line]
        assert len(rephrases)==5, "len(rephrases): {}, rephrases: {}".format(len(rephrases),rephrases)
        results.append({"original":original,"paraphrase":rephrases})

print("write {} lines".format(len(results)))
with open(output_file_path,"w",encoding="utf-8") as f:
    f.write(json.dumps(results,ensure_ascii=False,indent=4))