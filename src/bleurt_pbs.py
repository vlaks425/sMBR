from tqdm import tqdm
import argparse
import numpy as np
np.random.seed(1234)
# 参数解析
parser = argparse.ArgumentParser(description='Paired Bootstrap Resampling for Entity Recognition')
parser.add_argument('--hypo_files_scores', nargs='+', required=True, help='Paths to hypothesis files')
args = parser.parse_args()
# Paired Bootstrap Resampling 函数
def paired_bootstrap(baseline_scores, comparison_scores, n_iterations=10000,ratio=0.3):
    baseline_wins = 0
    comparison_wins = 0
    ties = 0
    baseline_accuracies = []
    comparison_accuracies = []

    for _ in range(n_iterations):
        sampled_indices = np.random.choice(len(baseline_scores), size=int(len(baseline_scores)*ratio), replace=True)
        sampled_baseline_scores = np.array(baseline_scores)[sampled_indices]
        sampled_comparison_scores = np.array(comparison_scores)[sampled_indices]

        baseline_accuracy = np.mean(sampled_baseline_scores)
        comparison_accuracy = np.mean(sampled_comparison_scores)

        baseline_accuracies.append(baseline_accuracy)
        comparison_accuracies.append(comparison_accuracy)

        if baseline_accuracy > comparison_accuracy:
            baseline_wins += 1
        elif comparison_accuracy > baseline_accuracy:
            comparison_wins += 1
        else:
            ties += 1
    # 判断显著性水平
    if baseline_wins > comparison_wins:
        print("Baseline is significantly better than Comparison with p-value: {}".format((1-baseline_wins/(n_iterations-ties))))
    elif comparison_wins > baseline_wins:
        print("Comparison is significantly better than Baseline with p-value: {}".format((1-comparison_wins/(n_iterations-ties))))
    else:
        print("Baseline and Comparison are not significantly different with p-value: {}".format(baseline_wins/(n_iterations-ties)))

    return {
        "baseline_wins": baseline_wins,
        "comparison_wins": comparison_wins,
        "ties": ties,
        "baseline_mean_accuracy": np.mean(baseline_accuracies),
        "comparison_mean_accuracy": np.mean(comparison_accuracies),
        "baseline_variance": np.var(baseline_accuracies),
        "comparison_variance": np.var(comparison_accuracies)
    }
# 处理假设文件
hypothesis_results = []
print("Processing Hypothesis Files listed below: {}".format(args.hypo_files_scores))
for hypo_file in args.hypo_files_scores:
    print(f"Processing Hypothesis File: {hypo_file}")
    with open(hypo_file, "r", encoding='utf-8') as f:
        hypothesis_lines = f.readlines()
        hypo_scores=[float(line.strip()) for line in hypothesis_lines]
    hypothesis_results.append(np.array(hypo_scores))
    print("the mean of {} is {}".format(hypo_file,np.mean(hypo_scores)))
# 计算Paired Bootstrap Resampling
print("Calculating Paired Bootstrap Resampling")
for comparison_system in range(1, len(hypothesis_results)):
    print("Comparing System {} with System 0".format(comparison_system))
    result=paired_bootstrap(hypothesis_results[0], hypothesis_results[comparison_system])
    print(result)

