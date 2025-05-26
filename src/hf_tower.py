import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--src_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--gpu', type=str, default="0", help='gpu id')
parser.add_argument('--num_hypotheses', type=int, required=True, default=1)
parser.add_argument('--max_generation_num', type=int, default=-1)
parser.add_argument("--lang_pair", type=str, default="zh2en")
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from transformers import set_seed
set_seed(114514)
from tqdm import tqdm
import torch
import numpy as np
from typing import  Optional, Tuple
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained(
    "Unbabel/TowerInstruct-13B-v0.1",
    padding_side="left",
    clean_up_tokenization_spaces=False,
    use_fast=False,
)
model = AutoModelForCausalLM.from_pretrained(
    "Unbabel/TowerInstruct-13B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
print("Model loaded.")


def compute_transition_scores(
        model,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        beam_indices=None,
    ) -> torch.Tensor:
        sequences=sequences.to("cpu")
        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
            beam_indices = beam_indices.expand(-1, len(scores))
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1).to("cpu")
        scores = scores.reshape(-1, model.config.vocab_size, scores.shape[-1])
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = scores.reshape(-1, scores.shape[-1])

        # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 6. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * model.config.vocab_size

        # 7. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 8. Compute scores
        transition_scores = scores.gather(0, indices)

        # 9. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0
        inf_mask = torch.isinf(transition_scores)
        transition_scores[inf_mask] = 0

        return transition_scores

def get_scores(model, outputs, output_length):
    if "sequences_scores" in outputs:
        return outputs["sequences_scores"].tolist()
    transition_scores = compute_transition_scores(model, outputs["sequences"],outputs["scores"]).cpu().numpy()
    reconstructed_scores = np.sum(transition_scores, axis=1) / (output_length)
    scores = reconstructed_scores.tolist()
    return scores

lp2prompt = {
    "zh2en": "Translate the following text from Chinese into English.\nChinese: ",
    "en2de": "Übersetzen Sie den folgenden Text vom Englischen ins Deutsche.\nEnglischen: ",
    "en2ru": "Переведите следующий текст с английского на русский.\nАнглийский: ",
}
lp2tgt = {
    "zh2en": "English",
    "en2de": "Deutsche",
    "en2ru": "Русский",
}

tokenizer.use_default_system_prompt = False
with open(args.src_file, "r", encoding="utf-8") as f:
    results = []
    scores=[]
    src_lines = f.readlines()
    src_lines = [
        {
            "role": "user",
            "content": f"{lp2prompt[args.lang_pair]}{l.strip()}\n{lp2tgt[args.lang_pair]}:",
        }
        for l in src_lines
        ]
    src_lines = [
        tokenizer.apply_chat_template(
            [p], tokenize=False, add_generation_prompt=True, add_special_tokens=False
        )
        for p in src_lines
    ]
    for line in tqdm(src_lines,total=len(src_lines),desc="generating hypotheses"):
        hypo_list = []
        scores_list = []
        with torch.no_grad():
            line_tok= tokenizer(line, return_tensors="pt").to("cuda")
            if args.max_generation_num > 0:
                while len(hypo_list)<args.num_hypotheses:
                    gen_num = min(
                        args.max_generation_num, args.num_hypotheses - len(hypo_list)
                    )
                    result = model.generate(
                        **line_tok,
                        max_new_tokens=256,
                        num_beams=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        num_return_sequences=gen_num,
                        do_sample=True,
                        top_k=0,
                        epsilon_cutoff=0.02,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    outputs = result["sequences"]
                    # remove prompt
                    outputs = outputs[:, line_tok["input_ids"].shape[-1] :]
                    hypo_list.extend(
                        tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    )
                    score = get_scores(model, result, result["sequences"].shape[-1])
                    scores_list.extend(score)
                    del result
                    del outputs
                    torch.cuda.empty_cache()
                assert len(hypo_list) == args.num_hypotheses, f"{len(hypo_list)} != {args.num_hypotheses}"
            else:
                result = model.generate(
                    **line_tok,
                    max_new_tokens=256,
                    num_beams=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_return_sequences=args.num_hypotheses,
                    do_sample=True,
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                    epsilon_cutoff=0.02,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                outputs = result["sequences"]
                # remove prompt
                outputs = outputs[:, line_tok["input_ids"].shape[-1] :]
                hypo_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                scores_list = get_scores(model, result, result["sequences"].shape[-1])
                del result
                del outputs

        hypo_list = [x for _, x in sorted(zip(scores_list, hypo_list), reverse=True)]
        scores_list.sort(reverse=True)
        results.extend(hypo_list)
        scores.extend(scores_list)
        del line_tok
        torch.cuda.empty_cache()

assert len(results) == len(src_lines) * args.num_hypotheses, (
    len(results),
    len(src_lines) * args.num_hypotheses,
)
assert len(scores) == len(src_lines) * args.num_hypotheses, (
    len(scores),
    len(src_lines) * args.num_hypotheses,
)
with open(args.output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line.replace("\n", "") + "\n")
with open(args.output_file+".score", "w", encoding="utf-8") as f:
    for line in scores:
        f.write(str(line)+"\n")
