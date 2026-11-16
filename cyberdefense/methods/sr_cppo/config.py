"""SR-CPPO recurrent policy, distributional cost critic and PID controller.
Default run settings. Command-line arguments select the dataset and run budget."""

RUN_SINGLE_SEED_QUICK_TEST = True

SINGLE_SEED = 42

MULTI_SEEDS = (42, 43, 44, 45, 46)

TOTAL_TRAIN_EPS = 1500

FROZEN_EVAL_EPS = 200

ENABLE_A1_FALLBACK_IN_EVAL = True

EVAL_ONLY_FROM_SAVED_CKPT = False
