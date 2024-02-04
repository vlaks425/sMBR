import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import math
from tqdm import tqdm
from transformers import XGLMTokenizer, XGLMForCausalLM
k_list=[5,16,32,64,80,96,128,160]
def return_perplexity(model, tokenizer, sentences,bs=4):
    #先创建一个空列表，用于存放每个batch的loss
    outputs = []
    for i in tqdm(range(len(sentences)//bs+1),desc="perplexity",total=len(sentences)//bs+1):
        batch_sentences=sentences[i*bs:(i+1)*bs]
        if len(batch_sentences)==0:
            break
        inputs = tokenizer(batch_sentences, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
        # 1トークンずらす
        shift_logits = output.logits[:, :-1, :].contiguous() # 確率
        shift_labels = inputs['input_ids'][:, 1:].contiguous() # 正解のトークンID
        shift_mask = inputs['attention_mask'][:, 1:].contiguous() # マスク
        batch_size, seq_len = shift_labels.shape
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none') # reduction='none'に
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(batch_size, seq_len)
        # shift_maskと積をとることで，pad_tokenに対する損失を無視する．
        # shift_mask.sum(dim=1)とすることで，各文のpad_tokenを除いた系列長が得られる
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        outputs.append(torch.exp(loss))
    outputs = torch.cat(outputs, dim=0)
    return outputs

print("loading model")
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
#gpu15
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B",cache_dir="/local/lyu",torch_dtype=torch.float16).to('cuda')
print("successfully load model")
print("compile model")
model = torch.compile(model)
print("successfully compile model")
tokenizer.pad_token = tokenizer.eos_token
for k in tqdm(k_list):
    with open("/raid/lyu/QEBT/test_dev/test_high_base/bs/"+str(k)+"/hypo_1","r",encoding="utf-8") as fin:
        sentences=[line.strip() for line in fin.readlines()]
        perplexity=return_perplexity(model,tokenizer,sentences)
        assert perplexity.shape[0]==len(sentences)
        with open("/raid/lyu/QEBT/test_dev/test_high_base/bs/"+str(k)+"/hypo_1.perplexity","w",encoding="utf-8") as fout:
            for p in perplexity:
                fout.write(str(p.item())+"\n")
for k in tqdm(k_list):
    with open("/raid/lyu/QEBT/test_dev/test_high_base/mbr_bs/output_"+str(k)+"/mbr.output","r",encoding="utf-8") as fin:
        sentences=[line.strip() for line in fin.readlines()]
        perplexity=return_perplexity(model,tokenizer,sentences)
        assert perplexity.shape[0]==len(sentences)
        with open("/raid/lyu/QEBT/test_dev/test_high_base/mbr_bs/output_"+str(k)+"/mbr.output.perplexity","w",encoding="utf-8") as fout:
            for p in perplexity:
                fout.write(str(p.item())+"\n")
for k in tqdm(k_list):
    with open("/raid/lyu/QEBT/test_dev/test_high_base/qe_bs/output_"+str(k)+"/qe.output","r",encoding="utf-8") as fin:
        sentences=[line.strip() for line in fin.readlines()]
        perplexity=return_perplexity(model,tokenizer,sentences)
        assert perplexity.shape[0]==len(sentences)
        with open("/raid/lyu/QEBT/test_dev/test_high_base/qe_bs/output_"+str(k)+"/qe.output.perplexity","w",encoding="utf-8") as fout:
            for p in perplexity:
                fout.write(str(p.item())+"\n")