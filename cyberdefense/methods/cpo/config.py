"""CPO. Reference: Achiam et al. (2017), Constrained Policy Optimization, ICML.
Default run settings. Command-line arguments select the dataset and run budget."""

RUN_SINGLE_SEED_QUICK_TEST = True

SINGLE_SEED = 42

MULTI_SEEDS = (42, 43, 44, 45, 46)

TOTAL_TRAIN_EPS = 1500

FROZEN_EVAL_EPS = 200
