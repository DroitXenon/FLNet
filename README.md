# DML Net

**Distributed Machine Learning with Adaptive Gradient Compression and Federated Knowledge Distillation for CNNs**

## Overview
DML Net is a research-driven project designed to optimize Distributed Machine Learning (DML) by combining Adaptive Gradient Compression (AGC) and Local Update Stochastic Gradient Descent (LU-SGD). The goal is to reduce communication overhead while maintaining model accuracy, making CNN-based distributed training more efficient.


## **Key Features**
- **Adaptive Gradient Compression (AGC):** Dynamically adjusts gradient compression using sparsification and quantization.
- **Local Update SGD (LU-SGD):** Reduces synchronization frequency by allowing local updates before aggregation.
- **Scalable Distributed Training:** Aims to optimize training performance across multiple devices.
- **PyTorch-Based Implementation:** Designed for flexibility and easy integration with PyTorch’s distributed training capabilities.

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/droitxenon/dml-net.git
   cd dml-net
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**
Run the training script with:
```bash
python scripts/train.py
```
Modify training parameters in `config.yaml` for different settings.

## **Team Members & Responsibilities**
- **Justin Wang** - Optimize performance and implementation of the combined algorithm
- **Eric Liu** - Implementation of LU-SGD
- **Oliver Liu** - Implementation of AGC
- **Anna An** - Design tests to measure communication overhead


