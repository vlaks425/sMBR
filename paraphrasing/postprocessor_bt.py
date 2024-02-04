import json
from tqdm import tqdm
k=16
#original_file_path="/raid/lyu/QEBT/test_dev/test_en2ru/bs/5/hypo_1"
#input_file_path="/raid/lyu/QEBT/test_dev/test_en2ru/paraphrasing_bt/hypo_16"
#output_file_path="/raid/lyu/QEBT/test_dev/test_en2ru/paraphrasing_bt/hypo_16.json"
original_file_path="/raid/lyu/QEBT/test_dev/test_en2de/bs/5/hypo_1"
input_file_path="/raid/lyu/QEBT/test_dev/test_en2de/paraphrasing_bt_oracle/hypo_16"
output_file_path="/raid/lyu/QEBT/test_dev/test_en2de/paraphrasing_bt_oracle/hypo_16.json"
inputs=[]
with open(input_file_path,"r",encoding="utf-8") as f:
    lines=f.readlines()
    for line in lines:
        inputs.append(line.strip())
with open(original_file_path,"r",encoding="utf-8") as f:
    originals=f.readlines()
    originals=[line.strip() for line in originals]
assert len(inputs)==(len(originals)*k)
results=[]
for i in tqdm(range(len(originals))):
    paraphrase=inputs[i*k:(i+1)*k]
    original=originals[i]
    results.append({"original":original,"paraphrase":paraphrase})
print("write {} lines".format(len(results)))
with open(output_file_path,"w",encoding="utf-8") as f:
    f.write(json.dumps(results,ensure_ascii=False,indent=4))
    