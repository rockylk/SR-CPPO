"""Upper-tail episodic-cost CVaR-PPO, adapted from Ying et al. (2022). Static CVaR: Rockafellar and Uryasev (2000).
Default run settings. Command-line arguments select the dataset and run budget."""

RUN_SINGLE_SEED_QUICK_TEST = True

SINGLE_SEED = 42

MULTI_SEEDS = (42, 43, 47, 45, 46)

TOTAL_TRAIN_EPS = 1500

FROZEN_EVAL_EPS = 200
