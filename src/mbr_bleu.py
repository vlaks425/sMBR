import argparse
import json
import numpy as np
parser = argparse.ArgumentParser(description='perform bleu based on mbr')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--utility_function', type=str, default="bleu")
parser.add_argument('--k', type=int, default=5)
args = parser.parse_args()