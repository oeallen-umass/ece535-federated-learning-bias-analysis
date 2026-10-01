# Bias Analysis in Federated Learning for Heterogeneous Sensors
Analysis of bias in federated learning across heterogeneous clients using FedAvg, TERM, and AFL.

## Team Members
- Will Lewis
- Oisin Allen

## Motivation
Federated learning allows multiple devices to collaboratively train a machine learning model without directly sharing their local data. However, differences in the data available to each client can cause the resulting global model to perform better for some groups than others. This project will investigate how different federated learning techniques affect these differences in model performance across clients.

## Design Goals
- Implement and evaluate multiple federated learning techniques.
- Compare FedAvg, TERM, and AFL under heterogeneous data distributions.
- Measure differences in model accuracy between client groups.
- Analyze how different federated learning techniques affect the variance in accuracy across clients.
- Visualize and compare the experimental results.

## System Overview
```text
Dataset
   ↓
Data Partitioning Across Clients
   ↓
Local Client Training
   ↓
Federated Learning Method
(FedAvg / TERM / AFL)
   ↓
Global Model
   ↓
Per-Client Accuracy Evaluation
   ↓
Bias and Variance Analysis
```

## Deliverables
- Working implementations of FedAvg, TERM, and AFL using provided or existing code.
- Federated learning experiments using a provided dataset.
- Evaluation of model accuracy across individual client groups.
- Comparison of accuracy variance across the different federated learning techniques.
- Graphs and visualizations showing experimental results.
- Analysis of how heterogeneous data distributions affect bias in federated learning.
- Final project demonstration and report.

## Software/Hardware Requirements
- No specialized hardware required
- Python
- PyTorch / machine learning libraries required by the provided implementations
- Google Colab or a computer with a CUDA-enabled GPU
- Git and GitHub
- CIFAR-10, FashionMNIST, or another provided dataset

## Team Responsibilities and Lead Roles
### Oisin Allen
- Setup
- Software
- Networking

### Will Lewis
- Research
- Algorithm Design
- Writing

## Project Timeline
### Phase 1 - Background Research and Setup
- Review federated learning concepts and the three selected techniques.
- Set up the development environment and provided code.
- Prepare the dataset.

### Phase 2 - Baseline Implementation
- Run and verify FedAvg.
- Establish baseline accuracy measurements.
- Set up per-client accuracy evaluation.

### Phase 3 - Additional FL Techniques
- Run and evaluate TERM.
- Run and evaluate AFL.
- Verify that all techniques use consistent experimental settings.

### Phase 4 - Experiments and Analysis
- Test different heterogeneous data distributions.
- Measure per-client accuracy.
- Calculate and compare accuracy variance.
- Generate graphs and visualizations.

### Phase 5 - Final Evaluation and Presentation
- Analyze results.
- Complete documentation and final report.
- Prepare final project demonstration and presentation.

## References
### Papers
1. **Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)**  
   https://arxiv.org/abs/1602.05629

2. **Tilted Empirical Risk Minimization (TERM)**  
   https://openreview.net/pdf?id=K5YasWXZT3O

3. **Agnostic Federated Learning (AFL)**  
   https://arxiv.org/pdf/1902.00146.pdf

### Code
- **FedAvg:** https://github.com/alexbie98/fedavg
- **TERM:** https://github.com/litian96/TERM
- **AFL:** https://github.com/YuichiNAGAO/agnostic_federated_learning

### Datasets
- **CIFAR-10:** https://www.kaggle.com/c/cifar-10/data
- **FashionMNIST:** https://github.com/zalandoresearch/fashion-mnist
