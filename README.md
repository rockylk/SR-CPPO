# SR-CPPO: Survivability-Oriented and Extreme Risk-Aware Autonomous Cyber Defense

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository provides the official implementation of **SR-CPPO (Safe-Reinforcement Constrained Proximal Policy Optimization)**, a survivability-oriented, extreme tail-risk-aware defense framework designed for autonomous cyber defense against Advanced Persistent Threats (APTs) in mission-critical enterprise networks.

## 📖 Overview

Modern enterprise and government networks operate under stringent Service Level Agreements (SLAs), prioritizing uninterrupted business operations. While existing deep reinforcement learning (DRL) methods offer adaptive defense, they are often risk-blind, leading to rare but catastrophic defense actions (e.g., severe service disruption). 

**SR-CPPO** addresses this by formulating the defense strategy as a Conditional Value-at-Risk (CVaR)-constrained Partially Observable Markov Decision Process (CPOMDP). It integrates:
- **Business-Aware Survivability Modeling**: Maps service criticality to network assets.
- **Tail-Risk-Aware Policy Optimization**: Suppresses extreme disruption risks via CVaR constraints.
- **Adaptive PID-Lagrangian Regulation**: Ensures stable and robust multi-objective optimization.
- **Dynamic Safety Fallback**: Prevents catastrophic defense failures and policy collapse.

## 📂 Repository Structure

The repository is modularized into three main components:

```text
.
├── SR-CPPO-code/         # Official SR-CPPO implementation
│   ├── src/              # Core modules (envs, models, agents, utils)
│   ├── main.py           # Training entry point
│   └── requirements.txt  
├── baseline/             # 5 State-of-the-art Baseline implementations
│   ├── CPO/              # Constrained Policy Optimization
│   ├── CPPO-PID/         # CPPO with PID-Lagrangian
│   ├── CVaR-PPO/         # PPO with CVaR penalties
│   ├── PPO/              # Unconstrained Proximal Policy Optimization
│   └── Rule-Based/       # Static Rule-Based Defense
└── Data/                 # High-fidelity telemetry datasets
    ├── optc_ot_sequence_benign2.npy
    └── optc_ot_sequence_malicious2.npy
```
> **Note**: Both `SR-CPPO-code` and the `baseline` implementations follow an identical, highly structured architecture (`envs`, `models`, `agents`, `utils`) for fair comparison and easy extensibility.

## ⚙️ Installation

We recommend using a virtual environment (e.g., Anaconda, `venv`, or `uv`).

```bash
# Clone the repository
git clone https://github.com/YourUsername/SR-CPPO.git
cd SR-CPPO

# Install dependencies (from the SR-CPPO-code directory)
cd SR-CPPO-code
pip install -r requirements.txt
```

*Key Dependencies:* `torch`, `numpy`, `scipy`

## 🚀 Usage

### 1. Running SR-CPPO
To train the SR-CPPO agent from scratch, navigate to the `SR-CPPO-code/` directory and execute the main entry point:

```bash
cd SR-CPPO-code
python main.py
```
The script will automatically load the telemetry datasets from the `Data/` directory and begin the training loop. Training logs and safety checkpoints will be generated dynamically.

### 2. Running Baselines
To evaluate or train any of the comparative baseline models, navigate to the specific baseline directory and run its `main.py` file. For example, to run the CPO baseline:

```bash
cd baseline/CPO
python main.py
```

## 📊 Dataset (OpTC Telemetry)
The `Data/` folder contains high-fidelity feature sequences extracted from the **OpTC (Operation Transparent Cyber)** dataset.
- `optc_ot_sequence_benign2.npy`: Benign background operational traffic.
- `optc_ot_sequence_malicious2.npy`: Adversarial threat patterns simulating APT attacks.

## 📝 Citation

If you find this code or our methodology useful in your research, please consider citing our paper:

```bibtex
@article{sr_cppo_2026,
  title={Survivability-Oriented and Extreme Risk-Aware Autonomous Cyber Defense via SR-CPPO},
  author={Kai  Liang, et al. },
  journal={IEEE Transactions on ...},
  year={2026}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
