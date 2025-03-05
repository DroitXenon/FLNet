# FLNet

**Federated Learning with Adaptive Gradient Compression and Federated Knowledge Distillation for CNNs**

## Overview
FLNet is a Federated Learning (FL) framework that integrates **Adaptive Gradient Compression (AGC)** and **Federated Knowledge Distillation** to optimize training efficiency for **Convolutional Neural Networks (CNNs)**. The goal is to reduce communication overhead while maintaining model accuracy in FL environments.

## Features
- **Federated Learning (FL)**: Decentralized model training without sharing raw data.
- **Adaptive Gradient Compression (AGC)**: Reduces communication cost by compressing gradients dynamically.
- **CNN Support**: Optimized for deep learning models like ResNet, MobileNet, and custom CNN architectures.
- **PyTorch Implementation**: Built on PyTorch for flexible deep learning experimentation.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/FLNet.git
cd FLNet
pip install -r requirements.txt
```

## Usage
### 1. Start the FL Server
Run the server to coordinate model updates from clients:
```bash
python server.py
```

### 2. Start FL Clients
Each client will train locally and send compressed updates to the server:
```bash
python client.py
```

### 3. Monitor Training
Use logging or visualization tools to track FL training progress.

## Project Structure
```
FLNet/
│── server.py        # Federated Learning server
│── client.py        # Federated Learning client
│── models/          # CNN model architectures
│── utils/           # Helper functions (data loading, compression, etc.)
│── experiments/     # Experimental setups and benchmarks
│── README.md        # Project documentation
│── requirements.txt # Python dependencies
```
