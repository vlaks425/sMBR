# sMBR

The official repository for our ACL2025 paper: **[Unveiling the Power of Source: Source-based Minimum Bayes Risk Decoding for Neural Machine Translation](https://aclanthology.org/2025.acl-long.149/)**.

This repository contains the official implementation of the Source-based Minimum Bayes Risk Paraphrase (sMBR-PP) decoding algorithm, designed to enhance the performance of Neural Machine Translation (NMT).

We demonstrate the application of sMBR-PP in NMT using the WMT2023 English-German translation task as an example. In this demonstration, the `Unbabel/TowerInstruct-13B-v0.1` as the NMT model.
Our experiments indicate that 48GB of GPU VRAM is sufficient to run the experiments with the parameters provided in the scripts.

## Setup and Usage

First, clone the repository:

```bash
git clone https://github.com/vlaks425/sMBR.git
cd sMBR
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

The necessary data has been pre-downloaded into the data/ directory for your convenience.
Models will be automatically downloaded to your Hugging Face cache directory (as configured in your environment).

## sMBR-PP Algorithm Overview
1. **Hypothesis Generation**: We generate multiple candidate translations (hypotheses) for each source sentence using a NMT model. In our case, we use the `Unbabel/TowerInstruct-13B-v0.1` (https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1) model to generate 128 hypotheses for each source sentence.
Run sMBR-PP:
Once hypotheses are generated, execute the `script/sMBR-PP_en2de_tower.sh` script to apply the sMBR-PP algorithm. This script takes the generated hypotheses as input and outputs the reranked translation results.

2. **Paraphrase Generation**: We fine-tune T5 models to generate paraphrases for the source segments. This step is crucial as it provides quasi-sources that are used in the sMBR-PP reranking process.
   - For English source segments, we use the model `lyu-boxuan/T5-sMBR-PP-EN`.
   - For Chinese source segments, we use the model `lyu-boxuan/T5-sMBR-PP-ZH`.
   - As demonstrated in our paper's appendix, leveraging advanced Large Language Models (LLMs) like GPT-4 for paraphrase generation can further enhance sMBR-PP performance. While we do not provide a GPT-4 implementation in this repository, we believe it is straightforward to implement by referring to our code.
3. **sMBR-PP Reranking**: The generated paraphrases (quasi-sources) and the hypotheses are then used for sMBR-PP reranking. We utilize the `Unbabel/wmt22-cometkiwi-da` model as the Quality Estimation (QE) model to score the hypotheses against the quasi-sources. The hypothesis with the highest sMBR-PP score is selected as the final translation.
## Running sMBR-PP
To run the sMBR-PP algorithm, run the following command:
 
### Generate Hypotheses
The following script uses epsilon samping (epsilon=0.02) to generate 128 hypotheses for each source.
```bash
bash script/gen_hypo_tower_en2de.sh
```
### sMBR-PP Reranking (with Paraphrase Generation) and Evaluation
```bash
bash script/sMBR-PP_en2de_tower.sh
```

The `script/sMBR-PP_en2de_tower.sh` script will automatically invoke `src/eval.sh` for evaluation.

Similarly, you can run standard QE reranking using `script/qe_reranking_en2de_tower.sh` and standard MBR decoding using `script/mbr_de_tower.sh` for comparison.

## Other Language Pairs
To experiment with other language pairs, such as English-Russian, modify the `src_file` and `tgt_file` variables in the scripts to point to the corresponding files under `data/wmt2023/`.

Note: For Chinese-English translation, ensure the `pp_model` variable in the script is set to `lyu-boxuan/T5-sMBR-PP-ZH` due to the use of a different paraphrase generation model.

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@inproceedings{lyu-etal-2025-unveiling,
    title = "Unveiling the Power of Source: Source-based Minimum {B}ayes Risk Decoding for Neural Machine Translation",
    author = "Lyu, Boxuan  and
      Kamigaito, Hidetaka  and
      Funakoshi, Kotaro  and
      Okumura, Manabu",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.149/",
    doi = "10.18653/v1/2025.acl-long.149",
    pages = "2976--2994",
    ISBN = "979-8-89176-251-0",
}
```
