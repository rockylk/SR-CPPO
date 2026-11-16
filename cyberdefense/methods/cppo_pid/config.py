"""CPPO-PID. Reference: Stooke et al. (2020), Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, ICML.
Default run settings. Command-line arguments select the dataset and run budget."""

RUN_SINGLE_SEED_QUICK_TEST = True

SINGLE_SEED = 42

MULTI_SEEDS = (42, 43, 45, 46, 47)

TOTAL_TRAIN_EPS = 1500

FROZEN_EVAL_EPS = 200
