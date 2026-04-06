# AutoML for Intracranial Aneurysm Rupture Prediction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the code, data, and models for the research paper: **"The Use of AutoML for Predicting Intracranial Aneurysm Rupture: A Comparison with Manually Curated Machine Learning Models"** by Brian H. Carlson B.S., B.S.E., Jason Carlson M.S., and Abhijith Bathini M.D.

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

Intracranial aneurysms affect approximately 3% of the global population and pose a significant risk of rupture, leading to subarachnoid hemorrhage with a mortality rate of around 50%. Accurate prediction of rupture risk is crucial for clinical decision-making regarding surgical intervention. Traditional models like the PHASES score have limitations due to their reliance on few variables and assumptions of linearity.

This study compares Automated Machine Learning (AutoML) approaches from Amazon Web Services (AWS) and Microsoft Azure with manually curated machine learning models using Bayesian Hyperparameter Optimization (BHO). The goal is to demonstrate that AutoML can produce effective predictive models with minimal human expertise, achieving performance comparable to expert-tuned models.

Key findings:
- AutoML models perform comparably to manually optimized models
- Tree-based algorithms consistently outperform other model types
- Azure's ExtremeRandomTrees achieved the highest AUC (0.7886)
- AutoML reduces overfitting risk and increases accessibility

## Background

### Intracranial Aneurysms and Rupture Risk

Intracranial aneurysms are focal dilations of cerebral arterial walls. While the annual rupture risk is relatively low (~0.76%), rupture often results in devastating consequences. Surgical interventions (clipping or endovascular coiling) carry their own risks, necessitating accurate rupture risk assessment.

### Limitations of Traditional Models

The PHASES score, a widely used clinical tool, predicts 5-year rupture risk based on:
- Geographical region
- Hypertension
- Age (≥70 years)
- Aneurysm size
- Prior subarachnoid hemorrhage
- Site of aneurysm

However, it assumes linear relationships and lacks many morphological and hemodynamic factors associated with rupture risk.

### Machine Learning Approaches

Machine learning can model complex, non-linear relationships using extensive feature sets. However, traditional ML requires significant expertise in hyperparameter tuning and model selection. AutoML automates these processes, making advanced ML accessible to non-experts.

## Data

The dataset consists of 112 intracranial aneurysms from 111 patients (44 ruptured, 68 unruptured), sourced from:
- University of Michigan
- Changhai Hospital (Shanghai)
- Aneurisk open-source repository

### Inclusion Criteria
- Sufficient data quality for CFD model establishment
- Aneurysm size: 4-25 mm
- No closely-spaced second aneurysms
- Anterior circulation location

### Features

**Anatomical Features (2):**
- Aneurysm location (MCA, ICA, BAS, ACA)
- Aneurysm type (Terminal/Lateral)

**Hemodynamic Parameters (9):**
- Systole STAWSS: Spatially/temporally averaged wall shear stress during peak systole
- Systole WSSMin/Max: Minimum/maximum wall shear stress during peak systole
- Mean OSI: Mean oscillatory shear index
- Std OSI: Standard deviation of oscillatory shear index
- TA LSA 2: Time-averaged low shear area (<0.4 Pa)
- TA LSA Std 2: Standard deviation of low shear area
- Systole TADVO: Time-averaged degree of vortex core overlap during systole
- Systole DVOStd: Standard deviation of vortex core overlap

**Morphological Features (13):**
- Bulbous shape indicator
- Aneurysm volume, height, sac maximum width
- Size ratios (height/width to parent vessel diameter)
- Parent vessel diameter
- Ostium diameters (min/max), area
- Aneurysm surface area
- Voronoi characteristic curve points (V1-V11)

### Data Processing

- 100 random train-test splits (80% train/validation, 20% test)
- Stratified sampling to maintain class distribution
- Feature engineering: initially tried square root/cube root transformations for area/volume features, but to make it easier to compare to other papers this was ommitted despite acheiving the best performance on final metrics.
- Additional derived features: OSI to ostium area ratio, sac width × size ratio

## Methods

### Local Models with Bayesian Hyperparameter Optimization (BHO)

Five models trained locally using Optuna:
- Logistic Regression (LR)
- Support Vector Classifier (SVC)
- Random Forest (RF)
- XGBoost
- Multilayer Perceptron (MLP)

**BHO Configuration:**
- MySQL database for trial sharing
- MedianPruner for early stopping
- TPESampler for hyperparameter suggestion
- Objective: Mean AUC across 5-fold CV minus 0.1 × overfitting penalty
- 75 trials per study, 16 parallel processes

### AWS AutoML (AutoGluon)

- TabularPredictor class on SageMaker
- ml.c4.8xlarge instance (36 vCPUs, 60GB RAM)
- Core principles: model diversity, bagging, stacked ensembling
- Models trained: CatBoost, NeuralNetFastAI, LightGBM, ExtraTrees, XGBoost, RandomForest, NeuralNetTorch
- "Best predictor" model selected per split

### Azure AutoML

- Azure Machine Learning Studio
- Auto-scaling compute cluster (Standard_DS11_v2, 0-50 nodes)
- Weighted AUC as primary metric
- Automatic featurization, BHO, probabilistic matrix factorization
- Models: LightGBM, XGBoost, ExtremeRandomTrees, RandomForest, LogisticRegression, Stack/Voting Ensembles

### Training Size Sweep Study

Additional analysis examining model performance across different training set sizes to assess scalability and robustness.

### Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score, AUC
- 95% confidence intervals using t-distribution
- Feature importance via permutation analysis
- ROC and Precision-Recall curves

## Results

### Performance Comparison

| Platform | Best Model | AUC | Accuracy | Recall | F1 Score |
|----------|------------|-----|----------|--------|----------|
| Local | Random Forest | 0.7843 | 0.7457 | 0.8407 | 0.8003 |
| AWS | ExtraTrees | 0.7767 | 0.7191 | 0.6701 | 0.6405 |
| Azure | ExtremeRandomTrees | **0.7886** | 0.7352 | 0.6425 | 0.6412 |

### Key Findings

1. **Comparable Performance**: AutoML models achieved performance similar to expert-tuned models and previous literature results.

2. **Tree-Based Superiority**: Random Forest, ExtraTrees, and ExtremeRandomTrees consistently outperformed other model types.

3. **Reduced Overfitting**: AutoML's automated processes may reduce overfitting compared to manual tuning.

4. **Feature Importance**: Consistent importance of ostium measurements, vessel diameter, and certain hemodynamic parameters across models.

5. **Training Size Effects**: Analysis shows how performance scales with dataset size, informing future studies.

### Feature Analysis

- Top correlated features: vessel diameter, ostium minimum diameter, ostium area
- Mutual information highlights non-linear relationships
- PCA analysis reveals variance structure (first 3 components: 26.4%, 16.0%, 9.2%)

## Repository Structure

```
├── data/
│   ├── README.md
│   └── newRuptureData/
│       ├── test/          # 100 test splits
│       └── train/         # 100 train splits
├── envs/
│   ├── aws_requirements.txt
│   └── local_env.yaml
├── models/
│   ├── aws_models.md
│   └── aneurysm_models_trained_locally/
│       ├── lr/, mlp/, rf/, svc/, xgb/
├── notebooks/
│   ├── aws/
│   │   ├── AneurysmTrainingSizeAblationStudy.ipynb
│   │   └── AWSRupturePredictionNotebook.ipynb
│   ├── azure/
│   │   └── AzureRupturePredictionNotebook.ipynb
│   └── local/
│       └── LocalRupturePredictionNotebookBHO.ipynb
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Conda or pip
- Git LFS (for large model files)

### Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/jcarlson212/AutoMLForAneurysmRupture.git
cd AutoMLForAneurysmRupture

# Install Git LFS
brew install git-lfs
git lfs install

# Set up local environment
conda env create -f ./envs/local_env.yaml
conda activate aneurysm-env

# For AWS reproduction
pip install -r ./envs/aws_requirements.txt
```

### Azure Setup

1. Create Azure Machine Learning workspace
2. Set up auto-scaling compute cluster (see Appendix C in paper)
3. Configure authentication and permissions

### AWS Setup

1. Launch SageMaker notebook instance
2. Use ml.c4.8xlarge or ml.c6i.8xlarge
3. Install AutoGluon and dependencies

## Usage

### Running Local Models

```bash
jupyter notebook notebooks/local/LocalRupturePredictionNotebookBHO.ipynb
```

### AWS Training Size Sweep

```bash
jupyter notebook notebooks/aws/AneurysmTrainingSizeAblationStudy.ipynb
```

### Azure AutoML

```bash
jupyter notebook notebooks/azure/AzureRupturePredictionNotebook.ipynb
```

### Data Processing

The repository includes pre-processed data splits. To regenerate:

```python
from sklearn.model_selection import train_test_split
# See Appendix A in paper for full preprocessing pipeline
```

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

### Git Configuration

```bash
# Increase post buffer for large notebooks
git config --global http.postBuffer 524288000  # 500MB

# Install Git LFS
brew install git-lfs
git lfs install
```

### Code Style

- Follow PEP 8 for Python code
- Use clear, descriptive variable names
- Include docstrings for functions
- Add comments for complex logic

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{carlson2024automl,
  title={The Use of AutoML for Predicting Intracranial Aneurysm Rupture: A Comparison with Manually Curated Machine Learning Models},
  author={Carlson, Brian H and Carlson, Jason and Bathini, Abhijith},
  journal={...},
  year={2024}
}
```

## Acknowledgments

- Dataset provided by: https://github.com/jjiang-mtu/IA-rupture-prediction
- Emory University Aneurisk Web for additional data
- OpenAI ChatGPT for assistance with manuscript preparation

---

**Disclaimer**: This research used AI tools for manuscript editing. Authors take full responsibility for the content.

