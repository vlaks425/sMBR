import argparse
parser = argparse.ArgumentParser(description='Process entities from texts.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the src file')
parser.add_argument('--score_file', type=str, required=True, help='Path to the score file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--num_hypotheses', type=int, required=True, default=1)
parser.add_argument('--k', type=int, required=True, default=1)

args = parser.parse_args()
input_file = args.input_file
score_file = args.score_file
output_file = args.output_file
num_hypotheses = args.num_hypotheses
k=args.k

with open(input_file, 'r',encoding='utf-8') as f:
    input_lines = f.readlines()
    input_lines = [line.strip() for line in input_lines]
with open(score_file, 'r',encoding='utf-8') as f:
    score_lines = f.readlines()
    score_lines = [float(line.strip()) for line in score_lines]
assert len(input_lines) == len(score_lines),print(len(input_lines),len(score_lines))
result = []

for i in range(0, len(input_lines), num_hypotheses):
    candidate_hypotheses = input_lines[i:i+k]
    if len(candidate_hypotheses)==0:
        break
    score=score_lines[i:i+k]
    best_hypothesis = candidate_hypotheses[score.index(max(score))]
    result.append(best_hypothesis)

assert len(result) == len(input_lines) / num_hypotheses

with open(output_file+"/qe.output", 'w',encoding='utf-8') as f:
    for line in result:
        f.write(line + '\n')