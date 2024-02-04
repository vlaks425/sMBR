from tqdm import tqdm
k=5
text_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt"
input_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt.QE_comet_scores"
output_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt.QE_comet_output"
max_score_output_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt.QE_comet_max_score"
with open(input_file, "r", encoding="utf-8") as fin:
    print("start to load scores")
    lines=fin.readlines()
    scores=[float(line.strip()) for line in lines]
    print("successfully load scores")
with open(text_file, "r", encoding="utf-8") as fin:
    print("start to load text")
    lines=fin.readlines()
    print("successfully load text")
results=[]
index=[]
#从k个假设的分数中，选择最高的分数作为该要提取的句子的分数
#并记录下索引，以便后续提取假设
for i in tqdm(range(len(scores)//k),desc="extract",total=len(scores)//k):
    cand_scores=scores[i*k:(i+1)*k]
    max_score=max(cand_scores)
    max_index=cand_scores.index(max_score)
    results.append(max_score)
    index.append(i*k+max_index)
assert len(results)==len(index)
assert len(results)==len(scores)//k
with open(output_file, "w", encoding="utf-8") as fout:
    for i in tqdm(index):
        fout.write(lines[i])
with open(max_score_output_file, "w", encoding="utf-8") as fout:
    for score in tqdm(results):
        fout.write(str(score)+"\n")