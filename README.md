# SGT: Sparse Growing Transformer

This repository contains the official code for our paper **"Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping"**.


[![arXiv](https://img.shields.io/badge/arXiv-2603.23998-b31b1b.svg)](https://arxiv.org/abs/2603.23998)

## Overview

Sparse Growing Transformer (SGT) is a training-time sparse depth allocation framework that progressively extends recurrence from deeper to shallower layers via targeted attention looping on informative heads. Unlike existing block-level recursive approaches that uniformly reapply entire layers, SGT induces structural sparsity by selectively increasing depth only for a small subset of high-entropy attention heads as training evolves.

## Installation

To install from source, run the following commands:

```bash
git clone https://github.com/YaoChen0203/Sparse-Growing-Transformer.git
cd Sparse-Growing-Transformer
pip install -e .[all]
```

## Data Preparation

The data preprocessing has been completed by the authors of [OLMo](https://github.com/allenai/OLMo). You need to prepare the preprocessed data for training.
Please refer to the [OLMo data preparation guide](https://github.com/allenai/OLMo/tree/main#pretraining) for instructions on downloading and preprocessing the data.
After preparing the data, modify the `data.paths` field in the config file to point to your local data paths.

## Pre-training

The following script shows how to launch SGT pre-training:

```bash
cd Sparse-Growing-Transformer
bash commands/run.sh
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{chen2026sparsegrowingtransformertrainingtime,
      title={Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping}, 
      author={Yao Chen and Yilong Chen and Yinqi Yang and Junyuan Shang and Zhenyu Zhang and Zefeng Zhang and Shuaiyi Nie and Shuohuan Wang and Yu Sun and Hua Wu and HaiFeng Wang and Tingwen Liu},
      year={2026},
      eprint={2603.23998},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.23998}, 
}
```

## Acknowledgments

This project is built upon [OLMo](https://github.com/allenai/OLMo) (v1.0) by the Allen Institute for AI (AI2). We thank the OLMo team for their open-source contribution, which made our research possible.

