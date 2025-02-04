# DVIB: Towards Robust Multimodal Recommender Systems via Variational Information Bottleneck Distillation

**By Wenkuan Zhao (zhaowk5@mail2.sysu.edu.cn)**

This repository contains the implementation of the paper *"DVIB: Towards Robust Multimodal Recommender Systems via Variational Information Bottleneck Distillation"*. Our paper has been accepted at the **2025 ACM Web Conference (WWW2025)**.

## Introduction

Multimodal recommender systems (MRS) aim to integrate various modalities to model user preferences and item characteristics more effectively, helping users discover items that align with their interests. While multimodal data can significantly improve performance, it also introduces challenges due to noise and information redundancy, which can affect the robustness of MRS.

Existing approaches typically address these issues separately, either by introducing perturbations at the input level for robust training to handle noise, or by designing complex network architectures to filter redundant information. In contrast, we propose the **DVIB framework**, which addresses both challenges simultaneously in a simple manner. Our approach involves shifting perturbations from the input layer to the hidden layers, combined with feature self-distillation, enabling noise mitigation and handling of redundancy without altering the original network architecture.

Additionally, we provide theoretical evidence demonstrating that the DVIB framework explicitly enhances model robustness and implicitly promotes an information bottleneck effect. This effect reduces redundant information during multimodal fusion and improves feature extraction quality. Extensive experiments show that DVIB consistently improves MRS performance across various datasets and model settings, complementing existing robust training methods. The DVIB framework represents a promising new paradigm for MRS.

## Quick Start

To run the code:

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Execute the main script:
   ```bash
   python main.py
   ```

You can modify the model and dataset by using the following commands:

- To use the `BM3` model with the `sports` dataset:
   ```bash
   python main.py --model BM3 --dataset sports
   ```

- To use the `VBPR` model with the `sports` dataset:
   ```bash
   python main.py --model VBPR --dataset sports
   ```

## Supported Models and Datasets

### Models:
- VBPR
- MMGCN
- GRCN
- BM3
- FREEDOM

For more details on the models, please check `src/models`. 

### Datasets:
- Amazon datasets: You can download the preprocessed `baby`, `sports`, and `clothing` datasets from Google Drive. 
- If you want to train models on other Amazon datasets, refer to the dataset processing tutorial.

## Checkpoints

*Todo.*

## Acknowledgments

We would like to express our gratitude to **enoche** for providing MMRec, which is used for multimodal recommendation tasks.


