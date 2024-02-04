import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--scores_file', type=str, required=True, help='Path to the scores file')
parser.add_argument('--text_file', type=str, required=True, help='Path to the text file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--metrice_name', type=str, required=True, help='metric name')
parser.add_argument('--k', type=int, required=True, help='k-best')
args = parser.parse_args()
with open(args.scores_file, "r", encoding="utf-8") as fscore,open(args.text_file, "r", encoding="utf-8") as ftext:
    scores=[float(line.strip()) for line in fscore.readlines()]
    texts=[line.strip() for line in ftext.readlines()]
    assert len(scores)==len(texts)
    best_scores=[]
    best_texts=[]
    for i in range(len(scores)//args.k):
        candidate_scores = scores[i*args.k:i*args.k+args.k]
        best_index = candidate_scores.index(max(candidate_scores))
        best_scores.append(candidate_scores[best_index])
        best_texts.append(texts[i*args.k+best_index])
    assert len(best_scores)==len(best_texts)
    assert len(best_scores)==len(texts)//args.k
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for text in best_texts:
            fout.write(text + "\n")
    with open(args.output_file+"best_{0}.score".format(args.metrice_name), "w", encoding="utf-8") as fout:
        for score in best_scores:
            fout.write(str(score) + "\n")
