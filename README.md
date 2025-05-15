# WSGraphCL: Weak-Strong Graph Contrastive Learning for Hyperspectral Image Classification

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/downloads/)

This repository contains the implementation of **WSGraphCL**, a weak-strong graph contrastive learning model designed for hyperspectral image (HSI) classification. The model is built to effectively handle noisy HSI data and requires only a limited amount of labeled data, making it particularly suitable for few-shot learning scenarios. For more details, please refer to our [paper]([https://ieeexplore.ieee.org/abstract/document/10988682]).

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

## Introduction

Hyperspectral images contain rich spectral and spatial information, making them valuable for various remote sensing applications. However, accurately classifying fine-grained land cover types in noisy HSIs remains challenging. Existing deep learning models often struggle with feature extraction from noisy data and require large labeled datasets for training.

To tackle these challenges, we propose **WSGraphCL**, a novel model that integrates contrastive learning and graph neural networks (GNNs). By leveraging weak-strong augmentations and filtering false negative pairs, our method stabilizes the pre-training process and learns robust representations. This enables effective HSI classification with minimal labeled data.

![Model Pipeline](modelpip.png)  <!-- Update this path with the actual path to your image -->

## Key Features

- **Graph Neural Network Architecture**: Utilizes a spectral-spatial adjacency matrix to construct K-hop subgraphs for effective feature extraction.
- **Contrastive Learning**: Pre-trains a graph-based encoder on unlabeled HSI data, reducing the need for extensive manual annotations.
- **Weak-Strong Augmentations**: Enhances the quality of learned representations by leveraging diverse augmentations during training.
- **Few-Shot Learning**: Demonstrates superior performance even with a handful of labeled samples, addressing the challenge of data scarcity.

## Installation

Clone the repository and install the required packages (Please note: requirements.txt is built on the Linux system, please find the matched PyTorch version to CUDA devices):

```bash
git clone https://github.com/zhu-xlab/WSGraphCL.git
cd WSGraphCL
pip install -r requirements.txt
```

## Usage

### Preprocessing

Prepare the Dataset: Download and prepare your hyperspectral [dataset](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). MDAS dataset can be downloaded from [here](https://mediatum.ub.tum.de/1657312). Ensure it is in the correct format as expected by the data loader.

### Training

To train the WSGraphCL model, use the following .ipynb files:
```bash
1. model/Indian_pines_mainfile.ipynb
2. model/Pavia_uni_mainfile.ipynb
3. model/MDAS_mainfile.ipynb
```
## Experiments

We conducted extensive experiments to validate the effectiveness of WSGraphCL under various few-shot scenarios. Our model consistently outperformed several baseline methods on benchmark HSI datasets.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@ARTICLE{10988682,
  author={Wang, Sirui and Braham, Nassim Ait Ali and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Weak–Strong Graph Contrastive Learning Neural Network for Hyperspectral Image Classification}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  keywords={Hyperspectral imaging;Contrastive learning;Feature extraction;Image classification;Training;Adaptation models;Earth;Data models;Noise;Image edge detection;Contrastive learning (CL);deep learning;graph neural networks (GNNs);hyperspectral image (HSI) classification;self-supervised learning (SSL)},
  doi={10.1109/TGRS.2025.3562261}}

```
## License


