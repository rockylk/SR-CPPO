# SR-CPPO: Safe-Reinforcement Constrained Proximal Policy Optimization

This repository contains the official PyTorch implementation of **SR-CPPO**, a survivability-oriented and extreme tail-risk-aware defense framework designed for Advanced Persistent Threats (APTs) in mission-critical networks.

## Features
- **CVaR-Constrained POMDP Formulation**: Optimizes business continuity and resilience under partial observability.
- **Tail-Risk-Aware Policy Optimization**: Supresses rare but catastrophic defense actions using Conditional Value-at-Risk (CVaR).
- **Adaptive PID-Lagrangian Regulation**: Dynamically adjusts penalty weights for robust multi-objective optimization.
- **Dynamic Safety Fallback**: Employs action gating and fallback heuristics to prevent extreme policy collapse.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure that the dataset files (`optc_ot_sequence_benign2.npy` and `optc_ot_sequence_malicious2.npy`) are placed in the `Data/` directory.

## Usage
Run the main script to start training the agent:
```bash
python main.py
```
