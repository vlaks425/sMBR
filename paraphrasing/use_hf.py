import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#gpu25
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir="/local/lyu/cache")

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0",cache_dir="/local/lyu/cache",use_flash_attention_2=True,torch_dtype=torch.float16)

prompt = "[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(0)

output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
