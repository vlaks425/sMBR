import argparse
import json
parser = argparse.ArgumentParser(description='perform sMBR')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--phrasing_file', type=str, required=True)
parser.add_argument('--original_file', type=str, required=True)
parser.add_argument('--k_input', type=int, default=5)
args = parser.parse_args()

input_file=args.input_file
output_file=args.output_file
phrasing_file=args.phrasing_file
original_file=args.original_file
k_input=args.k_input

with open(phrasing_file,"r",encoding="utf-8") as f:
    phrasing_results=json.load(f)
    original_1=[]
    for sample in phrasing_results:
        original_1.append(sample["original"])
    original_1=set(original_1)
with open(original_file,"r",encoding="utf-8") as f:
    original_2=[]
    for line in f.readlines():
        original_2.append(line.strip())
index=[]
for i in range(len(original_2)):
    if original_2[i] in original_1:
        index.append(i)
with open(input_file,"r",encoding="utf-8") as f:
    inputs=f.readlines()
    inputs=[line.strip() for line in inputs]
results=[]
for i in index:
    results.extend(inputs[i*k_input:(i+1)*k_input])
assert (len(results)//k_input)==len(phrasing_results),print(len(results)//k_input,len(phrasing_results))

with open(output_file,"w",encoding="utf-8") as f:
    for line in results:
        f.write(line+"\n")
        
print("successfully write {} lines".format(len(results)))