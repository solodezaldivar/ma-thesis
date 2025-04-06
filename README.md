# ma-thesis
# DVFU-WF Framework

**DVFU-WF** is a decentralized federated unlearning framework. It enables unlearning of specific shared models and evaluates the impact of unlearning using various metrics such as adversarial attacks, loss on training data and confidence on training and text data. 
The framework partitions data (e.g., MNIST, Fashion-MNIST) among multiple parties, trains local shared models, aggregates their weights to initialize global models, and then fine-tunes and evaluates the overall performance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Experiments and Evaluation](#experiments-and-evaluation)
- [License](#license)

## Overview

This project implements a federated learning and unlearning framework with the following key components:

- **Data Partitioning:**  
  Implements both round-robin and vertical (rotating) partitioning of datasets such as MNIST and Fashion-MNIST. This simulates IID, non-IID, and extreme non-IID cases.

- **Model Definitions:**  
    - **SharedModel:** A model that extracts intermediate (hidden) representations.
    - **GlobalModel:** A full model that aggregates shared representations from multiple parties and performs final classification.

- **Training and Aggregation:**  
  Shared models are trained in parallel on different partitions. Their weights are then aggregated (weighted average) to initialize the global model

- **Federated Rounds & Unlearning:**  
  The framework performs multiple federated rounds including a selective unlearning step. In the unlearning step, a target shared model is “forgotten” by removing its contribution and updating the global model accordingly.

- **Evaluation Metrics:**  
  The system evaluates model performance using:
    - Accuracy and F1-score.
    - Membership inference attack (MIA) loss.
    - Confidence score differences.
    - Adversarial membership inference attacks.

- **Resource Tracking & Visualization:**  
  Uses `psutil` to monitor CPU/GPU memory usage, training time before and after unlearning.

## Installation and Dependencies

The code requires the following libraries:

- Python 3.6+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [psutil](https://github.com/giampaolo/psutil)
- [matplotlib](https://matplotlib.org/)



```bash 
pip install torch torchvision numpy scikit-learn psutil matplotlib
```
Then to run the framework for MNIST and Fashion-MNIST
```bash
python `decentralized_vertical_federated_unlearning_wf.py`
```
and to run the framework for Bank-Marketing
```bash
python `dvfu-wl-bank.py`
```