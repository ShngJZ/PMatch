# PMatch

This repository contains the official implementation of the paper:  
**PMatch: Paired Masked Image Modeling for Dense Geometric Matching**, CVPR'23, [[arXiv]](https://arxiv.org/abs/2303.17342)

Authors: [Shengjie Zhu](https://shngjz.github.io/) and [Xiaoming Liu](https://cvlab.cse.msu.edu/)

## Installation

```bash
# Clone the repository
git clone https://github.com/shngjz/PMatchRelease.git
cd PMatchRelease

# Install dependencies
conda create -n pmatch python=3.9
conda activate pmatch
pip install -r requirements.txt
```

## Download Datasets

Download the benchmark MegaDepth and ScanNet datasets from Huggingface. Please ensure you agree to the licenses for each dataset.

```bash
git clone https://huggingface.co/datasets/shngjz/ce29d0e9486d476eb73163644b050222/
mv ce29d0e9486d476eb73163644b050222 TwoViewBenchmark
```

## Checkpoints

Download the pre-trained models from Huggingface:

| Model | Environment | Description | Link |
|-------|------------|-------------|------|
| PMatch-Indoor | Indoor | Trained on ScanNet | [pmatch_scannet.pth](https://huggingface.co/datasets/shngjz/ce29d0e9486d476eb73163644b050222/blob/main/checkpoints/pmatch_scannet.pth) |
| PMatch-Outdoor | Outdoor | Trained on MegaDepth | [pmatch_mega.pth](https://huggingface.co/datasets/shngjz/ce29d0e9486d476eb73163644b050222/blob/main/checkpoints/pmatch_mega.pth) |

Place the downloaded checkpoints in the `checkpoints` directory.

### Demo

Run a simple demo with your own images:

```bash
python PMatch/Benchmarks/demo.py
```

### Benchmarking

Evaluate PMatch on MegaDepth dataset:
```bash
python PMatch/Benchmarks/benchmark_pmatch_megadepth.py \
    --data_path /path/to/TwoViewBenchmark/megadepth_test_1500 \
    --checkpoints checkpoints/pmatch_mega.pth
```

Evaluate PMatch on ScanNet dataset:
```bash
python PMatch/Benchmarks/benchmark_pmatch_scannet.py \
    --data_path /path/to/TwoViewBenchmark/scannet_test_1500 \
    --checkpoints checkpoints/pmatch_scannet.pth
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{zhu2023pmatch,
    title={PMatch: Paired Masked Image Modeling for Dense Geometric Matching},
    author={Zhu, Shengjie and Liu, Xiaoming},
    booktitle={CVPR},
    year={2023}
}
```
