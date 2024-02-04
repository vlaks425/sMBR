from tqdm import tqdm
import json
import math
file = "/raid/lyu/QEBT/test_dev/test_en2de/as/400/hypo_400.score"
scores=[]
with open(file, "r",encoding="utf-8") as f:
    for line in tqdm(f):
        sample=json.loads(line.strip())
        scores.extend(sample)

with open(file+".2", "w",encoding="utf-8") as f:
    for line in tqdm(scores):
        #换底，换e为2
        line=math.log(line)
        line=2**line
        f.write(str(line)+"\n")