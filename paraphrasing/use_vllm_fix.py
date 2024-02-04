import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from vllm import LLM, SamplingParams
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


text_file_path="/raid/lyu/QEBT/en2de/wmt2020.en"
llm_output_file_path="/raid/lyu/QEBT/en2de/wmt2020.en2de.mixtral-8x7b-instruct-v0.1.4b-vllm-v0.0.1.json.old"
#text_file_path="/raid/lyu/en-ru/tests/generaltest2023.en-ru.src.en"
output_file_path="/raid/lyu/QEBT/en2de/wmt2020.en2de.mixtral-8x7b-instruct-v0.1.4b-vllm-v0.0.1.old.error_fix"
few_shot_demo_file_path="/home/lr/lyu/QEBT/paraphrasing/demonstration.txt"


with open(llm_output_file_path, "r", encoding="utf-8") as f:
    previous_results={}
    llm_output=json.load(f)
    for line in llm_output:
        previous_results[line["original"]]=line["paraphrase"]
print("len of previous results: {}".format(len(previous_results)))
with open(text_file_path, "r") as f:
    texts_=[]
    texts=[line.strip() for line in f.readlines()]
    for line in texts:
        if line not in previous_results:
            texts_.append(line)
    texts=texts_
print("read {} lines".format(len(texts)))
with open(few_shot_demo_file_path, "r", encoding="utf-8") as f:
    few_shot_demo=f.read()
print("len of few shot demo: {}".format(len(few_shot_demo)))
prompt=few_shot_demo+"\n\nOriginal: "
inputs=[]
for line in tqdm(texts,desc="make input",total=len(texts)):
    input=prompt+line+"\n"
    inputs.append("[INST] {} [/INST]".format(input))
results=[]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=2048, repetition_penalty=1.0,top_k=50,min_p=0.05)
print("Loading model...")
llm = LLM(model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",download_dir="/local/lyu/",dtype="float16",max_num_seqs=1)
print("Generating...")
outputs = llm.generate(inputs, sampling_params=sampling_params,prefix_pos=[len(prompt)]*len(inputs))
for i in range(len(outputs)):
    prompt = texts[i]
    generated_text = outputs[i].outputs[0].text
    if is_output_valid(generated_text):
        results.append({"original":prompt,"paraphrase":generated_text})
    else:
        _iter=1
        while True:
            print("Invalid output! Try again! (iter {})".format(_iter))
            output = llm.generate([inputs[i]], sampling_params=sampling_params)
            generated_text=output[0].outputs[0].text
            if is_output_valid(generated_text):
                results.append({"original":prompt,"paraphrase":generated_text})
                break
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in results:
        f.write(json.dumps(line, ensure_ascii=False))
        f.write("\n")