# ARGO: Interactive Data Governance System via Hierarchical Reinforcement Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/VLDB-2024-orange.svg" alt="VLDB">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **ARGO** is an interactive system for adaptive data governance via hierarchical reinforcement learning (HRL). It models data governance for machine learning as a sequential decision-making process, dynamically coordinating multiple operations including feature repairing, relabeling, augmentation, and sample removal.

This repository contains the implementation of ARGO, accepted at **VLDB 2024**.

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Citation](#-citation)
- [License](#-license)

## 📋 Overview

Training data quality is a key determinant of machine learning performance. Real-world datasets often suffer from:

- **Noisy Labels**: Incorrectly labeled samples (e.g., spam marked as non-spam)
- **Missing/Incorrect Features**: Invalid attribute values (e.g., `age = -5`)
- **Imbalanced Distributions**: Skewed class ratios affecting model fairness

Existing methods typically address these issues independently with predefined pipelines, making them difficult to generalize across diverse data modalities and evolving data characteristics.

**ARGO** addresses these challenges through **hierarchical reinforcement learning**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARGO System                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    High-Level: PPO Agent                            │     │
│  │                                                                      │     │
│  │   State St = [n_samples, class_ratio, accuracy, noise_ratio,      │     │
│  │                recent_rewards, feature_stats, action_counts]       │     │
│  │                                                                      │     │
│  │   Action At ∈ {repair_features, relabel, augment, remove, skip}    │     │
│  └─────────────────────────────┬───────────────────────────────────────┘     │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    Low-Level: Multi-Selector                         │     │
│  │                                                                      │     │
│  │   Selector 0: Feature Noise Detector → Repair target samples       │     │
│  │   Selector 1: Label Noise Detector → Flip target labels             │     │
│  │   Selector 2: Sample Removal Selector → Delete low-quality samples  │     │
│  │   Selector 3: Sample Augmentation Selector → Add synthetic samples  │     │
│  └─────────────────────────────┬───────────────────────────────────────┘     │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    Data Governance Environment                       │     │
│  │                                                                      │     │
│  │   • State: Dataset statistics & model performance metrics           │     │
│  │   • Action: Apply governance operations to dataset                   │     │
│  │   • Reward: Downstream model accuracy + efficiency penalties        │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Key Contributions

| Feature | Description |
|---------|-------------|
| **Hierarchical RL** | High-level PPO agent decides *what* operation; low-level selectors decide *which* samples |
| **Multi-Operation Coordination** | Dynamically sequences feature repairing, relabeling, augmentation, and removal |
| **Human-in-the-Loop** | Incorporates user feedback as rewards and supervision signals |
| **Interactive GUI** | Real-time visualization of dataset quality, model performance, and decision trajectories |
| **Plug-and-Play Modules** | Supports extensible noise detectors (IDE, ED2-RPT) and generators (SMOTE, VAE, GAN, LLM) |

## 🏗️ Architecture

### System Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│   ARGO       │────▶│ Clean Data   │
│  (Noisy)     │     │   (HRL)      │     │ (High Quality)│
└──────────────┘     └──────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   ML Model   │
                    │ (Improved)   │
                    └──────────────┘
```

### Agent Design

| Component | Role |
|-----------|------|
| **PPO Agent** | High-level policy for operation selection based on dataset state |
| **SelectorNet** | Sample-level selection with auxiliary prediction heads |
| **Noise Detectors** | Pre-training modules for initialization |
| **Data Generators** | Synthetic data augmentation  |

### Supported Operations

| Operation | Description |
|-----------|-------------|
| `modify_features` | Detect and repair noisy/missing feature values |
| `modify_labels` | Flip incorrectly labeled samples |
| `add_samples` | Add high-quality synthetic samples |
| `delete_samples` | Delete low-quality or harmful samples |

## 📥 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
git clone https://github.com/yourusername/ARGO.git
cd ARGO
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
flask>=3.0.0
```

## 🚀 Quick Start

### Training

```bash
python train_multi_selector_v2.py
```

### Interactive Demo

```bash
cd demo
python app.py
# Open http://localhost:5000
```

### Evaluation

```bash
python evaluate.py  
```

## 📁 Project Structure

```
ARGO/
├── ARGO/                                 # Core framework
│   ├── agents/                           # RL agents
│   │   ├── ppo_agent.py                  # High-level PPO agent
│   │   ├── multi_selector_agent.py       # Low-level multi-selector
│   │   └── networks.py                   # Neural network architectures
│   ├── env/                              # Environment
│   │   ├── data_cleaning_env.py          # Core data governance env
│   │   ├── noise_detectors/              # Noise detection modules  
│   │   └── data_generators/              # Data augmentation 
│   └── __init__.py
├── demo/                                 # Interactive web demo
│   ├── app.py                            # Flask application
│   ├── templates/                        # HTML templates
│   └── static/                           # CSS/JS assets
├── train_multi_selector_v2.py            # Main training script
├── config.py                             # Configuration
├── evaluate.py                           # Evaluation script
├── requirements.txt
└── README.md
```

## 📊 Experimental Results

We evaluate ARGO on three datasets spanning different data modalities:

### Performance Comparison

| Dataset | Type | Initial Accuracy | Final Accuracy | Improvement |
|---------|------|------------------|----------------|-------------|
| Adult | Tabular | ~70% | ~73% | +3% |
| CIFAR-10 | Image | ~71% | ~74% | +3% |
| IMDB | Text | ~60% | ~66% | +6% |

 

### Key Observations

1.Multi-modal Generalization: ARGO is effective across tabular, text, and image data modalities.
2.Progressive Learning: The accuracy steadily improves as training progresses, demonstrating the effectiveness of HRL.
 

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Environment
state_dim = 22          # Dataset state vector dimension
n_actions = 5           # Number of governance operations

# PPO Hyperparameters
gamma = 0.99            # Discount factor
gae_lambda = 0.95       # GAE parameter
ppo_epochs = 10         # PPO update epochs
clip_ratio = 0.2        # PPO clipping ratio
 


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ If you find this work useful, please cite our paper and give this repository a star!
