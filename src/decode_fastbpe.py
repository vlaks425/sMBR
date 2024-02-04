import argparse
from sacremoses import MosesDetokenizer
from tqdm import tqdm
parser = argparse.ArgumentParser(description='detokenize')
parser.add_argument('--input', type=str, required=True, help='Path to the input file')
parser.add_argument('--output', type=str, required=True, help='Path to the output file')
parser.add_argument('--lang', type=str, required=True, help='Language')
args = parser.parse_args()
md = MosesDetokenizer(lang=args.lang)
input_file=args.input
output_file=args.output

bpe_symbol = "@@ "

def detokenize(line):
    return (line + " ").replace(bpe_symbol, "").rstrip()


def mosesdetokenize(line):
    return md.detokenize(line.split())

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    lines=fin.readlines()
    for line in tqdm(lines,desc="detokenize",total=len(lines)):
        line=line.strip()
        line=detokenize(line)
        line=mosesdetokenize(line)
        fout.write(line+"\n")
