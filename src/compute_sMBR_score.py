import argparse
import json
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser(description='perform sMBR')
parser.add_argument('--qe_score_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--phrasing_qe_score_file', type=str, required=True)
parser.add_argument('--hypo_file', type=str, required=True)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--mean', action='store_true', help='use mean score')
args = parser.parse_args()

qe_score_file=args.qe_score_file
output_file=args.output_file
phrasing_qe_score_file=args.phrasing_qe_score_file
hypo_file=args.hypo_file
k=args.k

with open(qe_score_file,"r",encoding="utf-8") as f:
    qe_scores=[float(line.strip()) for line in f.readlines()]
with open(phrasing_qe_score_file,"r",encoding="utf-8") as f:
    phrasing_qe_lines=f.readlines()
    phrasing_qe_scores=[]
    for line in phrasing_qe_lines:
        phrasing_qe_scores.append(np.array(json.loads(line.strip())))
assert len(qe_scores)==len(phrasing_qe_scores)

sMBR_scores=[]
for i in range(len(qe_scores)):
    score=phrasing_qe_scores[i]
    if args.mean:
        mean_score=np.mean(np.append(score,qe_scores[i]))
        sMBR_scores.append(mean_score)
    else:
        mean_score=np.mean(phrasing_qe_scores[i])
        sMBR_scores.append(args.alpha*qe_scores[i]+(1-args.alpha)*mean_score)
assert len(sMBR_scores)==len(qe_scores)
with open(hypo_file,"r",encoding="utf-8") as f:
    hypo=[line.strip() for line in f.readlines()]
assert len(hypo)==len(qe_scores)

with open(output_file,"w",encoding="utf-8") as f:
    for i in range(len(hypo)//k):
        scores=sMBR_scores[i*k:(i+1)*k]
        best_index=scores.index(max(scores))
        f.write(hypo[i*k+best_index]+"\n")
with open(output_file+".score","w",encoding="utf-8") as f:
    for score in sMBR_scores:
        f.write(str(score)+"\n")
    