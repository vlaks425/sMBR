import argparse
import json
parser = argparse.ArgumentParser(description="extractor_jsonl4metricx")
parser.add_argument(
    "--input_file", type=str, required=True, help="Path to the input file"
)
parser.add_argument(
    "--output_file", type=str, required=True, help="Path to the output file"
)
args = parser.parse_args()
with open(args.input_file, "r", encoding="utf-8") as f:
    tmp=f.readlines()
    src_hypo=[json.loads(line.strip()) for line in tmp]
scores=[]
for data in src_hypo:
    scores.append(data["prediction"])
with open(args.output_file, "w", encoding="utf-8") as f:
    for line in scores:
        f.write(str(line) + "\n")