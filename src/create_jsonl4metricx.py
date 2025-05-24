import argparse
import json
parser = argparse.ArgumentParser(description="Convert jsonl to metricx.")
parser.add_argument(
    "--src_file", type=str, required=True, help="Path to the source file"
)
parser.add_argument(
    "--hypo_file", type=str, required=True, help="Path to the hypothesis file"
)
parser.add_argument(
    "--output_file", type=str, required=True, help="Path to the output file"
)
args = parser.parse_args()
with open(args.src_file, "r", encoding="utf-8") as f:
    src = f.readlines()
with open(args.hypo_file, "r", encoding="utf-8") as f:
    hypo = f.readlines()
assert len(src) == len(hypo),f"Length mismatch: {len(src)} vs {len(hypo)}"
data = []
for i in range(len(src)):
    data.append({"source": src[i].strip(), "hypothesis": hypo[i].strip()})
with open(args.output_file, "w", encoding="utf-8") as f:
    for line in data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")