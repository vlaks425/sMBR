from tqdm import tqdm
import json
from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="/cache01/lyu/mixtral-8x7b-instruct-v0.1.Q6_K.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=16,
  n_gpu_layers=-1,
  use_mmap=True
)

text_file_path="/raid/lyu/QEBT/en2de/wmt2020.en"
output_file_path="/local/lyu/wmt2020.en2de.mixtral-8x7b-instruct-v0.1.Q6_K"
few_shot_demo_file_path="/home/lr/lyu/QEBT/paraphrasing/demonstration.txt"


with open(text_file_path, "r") as f:
    texts=[line.strip() for line in f.readlines()]
print("read {} lines".format(len(texts)))
with open(few_shot_demo_file_path, "r", encoding="utf-8") as f:
    few_shot_demo=f.read()
print("len of few shot demo: {}".format(len(few_shot_demo)))
prompt=few_shot_demo+"\n\nOriginal: "
results=[]
for line in tqdm(texts,desc="generating paraphrases",total=len(texts)):
    result={}
    output=llm("[INST] {} [/INST]".format(prompt+line+"\n"), max_tokens=512,stop=["</s>"],echo=False,repeat_penalty=1.0,top_k=50,top_p=0.95)
    result["original"]=line
    result["paraphrase"]=output["choices"][0]["text"]
    results.append(result)
print(results)
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in results:
        f.write(json.dumps(line, ensure_ascii=False, indent=4))
        f.write("\n")