import sacrebleu
import argparse
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

# 函数来处理每个hypothesis group
def process_group(group):
    k = len(group)
    bleu = sacrebleu.BLEU(smooth_method="add-k", effective_order=True)
    mbr_scores = []
    bleu_scores = np.zeros((k, k))
    for j, hypo in enumerate(group):
        for l, ref in enumerate(group):
            if j == l:
                continue
            bleu_scores[j][l] = bleu.sentence_score(hypo, [ref]).score
    for j in range(k):
        mbr_scores.append(np.mean(bleu_scores[j]))
    best_mbr_score = max(mbr_scores)
    best_mbr_hypo = group[mbr_scores.index(best_mbr_score)]
    return best_mbr_score, best_mbr_hypo


def process_group_zh(group):
    k = len(group)
    bleu = sacrebleu.BLEU(smooth_method="add-k", effective_order=True, tokenize="zh")
    mbr_scores = []
    bleu_scores = np.zeros((k, k))
    for j, hypo in enumerate(group):
        for l, ref in enumerate(group):
            if j == l:
                continue
            bleu_scores[j][l] = bleu.sentence_score(hypo, [ref]).score
    for j in range(k):
        mbr_scores.append(np.mean(bleu_scores[j]))
    best_mbr_score = max(mbr_scores)
    best_mbr_hypo = group[mbr_scores.index(best_mbr_score)]
    return best_mbr_score, best_mbr_hypo


def main():
    parser = argparse.ArgumentParser(description='Process entities from texts.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--hypo_file', type=str, required=True, help='Path to the hypothese file')
    parser.add_argument('--k', type=int, default=5, help='k-best')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--is_zh', action='store_true', help='Whether the input is Chinese')
    args = parser.parse_args()
    is_zh = True if args.is_zh else False
    with open(args.hypo_file, "r", encoding='utf-8') as f:
        hypo_lines = [line.strip() for line in f.readlines()]
    
    num_groups = len(hypo_lines) // args.k
    assert len(hypo_lines) % args.k == 0, "Number of lines in the hypothesis file should be a multiple of k"

    # 将数据分组
    groups = [hypo_lines[i*args.k:(i+1)*args.k] for i in range(num_groups)]

    # 使用多进程
    if is_zh:
        with Pool(args.num_workers) as p:
            results = list(tqdm(p.imap(process_group_zh, groups), total=len(groups)))
    else:
        with Pool(args.num_workers) as p:
            results = list(tqdm(p.imap(process_group, groups), total=len(groups)))

    # 写入结果
    with open(args.output_file, "w", encoding='utf-8') as f, \
         open(args.output_file+".scores", "w", encoding='utf-8') as f_scores:
        for score, hypo in results:
            f.write(hypo + "\n")
            f_scores.write(str(score) + "\n")

if __name__ == "__main__":
    main()

    
    
    