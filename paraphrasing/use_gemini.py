import google.generativeai as genai
from tqdm import tqdm
import json
#from google.colab import userdata

GOOGLE_API_KEY='AIzaSyCTpCVN3IBGyiuRrUXaKaRcNN1VRG02xzE'

genai.configure(api_key=GOOGLE_API_KEY)
generationconfig={
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "temperature": 0,
        "top_p": 1,
        "top_k":1
    }

safety_settings_NONE=[
        { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

model = genai.GenerativeModel('gemini-pro')
text_file_path="/raid/lyu/QEBT/en2de/wmt2020.en"
output_file_path="/local/lyu/wmt2020.en2de.gemini_pro"
few_shot_demo_file_path="/home/lr/lyu/QEBT/paraphrasing/demonstration.txt"


with open(text_file_path, "r") as f:
    texts=[line.strip() for line in f.readlines()]
texts=[texts[7]]
print("read {} lines".format(len(texts)))
with open(few_shot_demo_file_path, "r", encoding="utf-8") as f:
    few_shot_demo=f.read()
print("len of few shot demo: {}".format(len(few_shot_demo)))

prompt=few_shot_demo+"\n\nOriginal: "
results=[]
for line in tqdm(texts,desc="generating paraphrases",total=len(texts)):
    result={}
    input_text=prompt+line+"\n"
    
    response = model.generate_content(input_text,generation_config=generationconfig,safety_settings=safety_settings_NONE)
    print(response.text)
    exit()
    result["original"]=line
    #result["paraphrase"]=response.text
    results.append(result)
print(results)
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in results:
        f.write(json.dumps(line, ensure_ascii=False, indent=4))
        f.write("\n")