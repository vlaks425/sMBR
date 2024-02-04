
import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')
parser.add_argument('--k', type=int, default=5, help='k-best')
parser.add_argument('--decoding_method', type=str, default="bs", help='decoding method')
parser.add_argument('--num_hypotheses', type=int, required=True, default=1)
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from typing import  Optional, Tuple
from transformers import set_seed
set_seed(114514)
from tqdm import tqdm
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-en-de"
decoding_method = args.decoding_method
k=args.k
num_hypotheses=args.num_hypotheses
src_file = args.src_file
output_path = args.output_file
#如果不存在output_path，则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_path + decoding_method):
    os.makedirs(output_path + decoding_method)
if not os.path.exists(output_path + decoding_method+"/"+str(num_hypotheses)):
    os.makedirs(output_path + decoding_method+"/"+str(num_hypotheses))
output_file = output_path + decoding_method+"/"+str(num_hypotheses)+"/hypo_"+str(num_hypotheses)

batch_size = 1
print("Using model: %s" % mname)
tokenizer = FSMTTokenizer.from_pretrained(mname,cache_dir="/cache01/lyu/checkpoints")
model = FSMTForConditionalGeneration.from_pretrained(mname,cache_dir="/cache01/lyu/checkpoints").to("cuda")
print("Model loaded.")

def compute_scores(sequences: torch.Tensor,scores: Tuple[torch.Tensor],beam_indices: Optional[torch.Tensor] = None,normalize_logits: bool = False,vocab_size: Optional[int] = None) -> torch.Tensor:
    #convert to cpu tensor
    sequences=sequences.to("cpu")
    tmp_scores=()
    for score in scores:
        tmp_scores+=(score.to("cpu"),)
    scores=tmp_scores
    del tmp_scores
    
    # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
    # to a beam search approach were the first (and only) beam is always selected
    if beam_indices is None:
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
        beam_indices = beam_indices.expand(-1, len(scores))

    # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
    # seq_len - input_length
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

    # 3. Optionally normalize the logits (across the vocab dimension)
    if normalize_logits:
        scores = scores.reshape(-1, vocab_size, scores.shape[-1])
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = scores.reshape(-1, scores.shape[-1])

    # 4. cut beam_indices to longest beam length
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices.clone()[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]

    # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
    beam_indices[beam_indices_mask] = 0

    # 6. multiply beam_indices with vocab size to gather correctly from scores
    beam_sequence_indices = beam_indices * vocab_size

    # 7. Define which indices contributed to scores
    cut_idx = sequences.shape[-1] - max_beam_length
    indices = sequences[:, cut_idx:] + beam_sequence_indices

    # 8. Compute scores
    transition_scores = scores.gather(0, indices)
    # 9. Mask out transition_scores of beams that stopped early
    transition_scores[beam_indices_mask] = 0
    
    # 10. cut off -inf scores
    inf_mask = torch.isinf(transition_scores)
    transition_scores[inf_mask] = 0
    
    seq_score=transition_scores.sum(axis=1)
    # 11.devide by length
    seq_score=seq_score/(sequences.shape[-1]-cut_idx)
    seq_score=torch.pow(seq_score,2)

    return seq_score


with open(src_file, "r", encoding="utf-8") as f:
    results = []
    scores=[]
    src_lines = f.readlines()
    src_lines = [line.strip() for line in src_lines]
    for line in tqdm(src_lines,total=len(src_lines),desc="预测"):
        with torch.no_grad():
            line_tok= tokenizer(line, return_tensors="pt").to("cuda")
            outputs = model.generate(**line_tok,max_length=200,num_beams=k,output_scores=True,return_dict_in_generate=True,num_return_sequences=num_hypotheses,do_sample=True,top_k=0)
            result = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            results.extend(result)
            transition_scores=compute_scores(outputs["sequences"],outputs["scores"],vocab_size=model.config.tgt_vocab_size,normalize_logits=True)
            scores.extend(transition_scores.tolist())
        del line_tok
        del result
        del outputs
        del transition_scores
        torch.cuda.empty_cache()
    #assert len(results) == len(src_lines),f"len(results)={len(results)},len(src_lines)={len(src_lines)}"
with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line+"\n")
with open(output_file+".score", "w", encoding="utf-8") as f:
    for line in scores:
        f.write(str(line)+"\n")
