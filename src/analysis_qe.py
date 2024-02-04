from tqdm import tqdm
import matplotlib.pyplot as plt
k=5
text_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt"
input_file="/raid/lyu/QEBT/bt_data/en-zh/news_crawl_zh_300k_hypo_5.bt.QE_comet_scores"
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
analysis={}
for i in tqdm(range(len(scores)//k),desc="extract",total=len(scores)//k):
    cand_scores=scores[i*k:(i+1)*k]
    max_score=max(cand_scores)
    max_index=cand_scores.index(max_score)
    results.append(max_score)
    index.append(i*k+max_index)
    if max_index not in analysis:
        analysis[max_index]=1
    else:
        analysis[max_index]+=1
#打印百分比和保存图片
#sort
analysis=dict(sorted(analysis.items(),key=lambda item:item[1],reverse=True))
print("analysis:")
for key in analysis:
    print(key,analysis[key]/(len(scores)//k))
plt.hist(results,bins=100)
plt.savefig("/home/lr/lyu/QEBT/debug/hist.png")
