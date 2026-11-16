# SR-CPPO

[Previous code snapshot](https://github.com/rockylk/SR-CPPO/tree/archive/pre-update-20261116)

Modular implementations of SR-CPPO, five comparison methods, and component ablations on OpTC. The main method uses the same training and evaluation code for OpTC and DAPT 2020, with dataset selection through command-line arguments.

[中文说明](README.zh-CN.md)

## Project structure

```text
SR-CPPO/
├── run.py                         # Entry point for SR-CPPO and baselines
├── run_ablation.py                # Entry point for OpTC ablations
├── requirements.txt
├── data/
│   ├── optc/                      # OpTC feature arrays
│   └── dapt2020/                  # DAPT 2020 features and optional labels
└── cyberdefense/
    ├── common/                    # Data validation, paths, and logging
    ├── methods/
    │   ├── sr_cppo/
    │   ├── ppo/
    │   ├── cpo/
    │   ├── cppo_pid/
    │   ├── cvar_ppo/
    │   └── rule_based/
    └── ablation/                  # Component variants, training, evaluation, and resume support
```

Each learning method is organized into `models.py`, `environment.py`, `risk.py`, `agent.py`, `training.py`, `evaluation.py`, and `runner.py`. The `runtime.py` module handles random seeds and logging. Rule-Based has no training module because it does not train a network. For ablations, `io.py` manages checkpoints, and `base_agent.py` contains the SR-CPPO core used by the variants.

## Environment

Use Python 3.10–3.12. Activate your PyTorch environment and run the following command from this directory:

```bash
python -m pip install -r requirements.txt
```

An existing GPU-enabled PyTorch installation can be used directly. The entry point selects the GPU when CUDA is available. Use `--device cpu` to run on the CPU and `--threads 1` to limit CPU threads. Run metadata records the Python, NumPy, and PyTorch versions and the device used.

## Data

| Dataset | Benign feature file | Malicious feature file | Shapes: benign / malicious |
|---|---|---|---|
| OpTC | `data/optc/optc_ot_sequence_benign2.npy` | `data/optc/optc_ot_sequence_malicious2.npy` | 24,988 × 16 / 3,338 × 16 |
| DAPT 2020 | `data/dapt2020/dapt_ot_sequence_benign2.npy` | `data/dapt2020/dapt_ot_sequence_malicious2.npy` | 17,018 × 16 / 1,026 × 16 |

Training uses the 16-dimensional arrays produced by the existing extraction pipeline. The entry points do not re-extract or transform these inputs. The `dapt_ot_labels_*2.npy` files contain accompanying labels; the training environment reads the feature files listed above.

Default data paths are resolved relative to the project directory, independently of the terminal's current working directory. To use data stored elsewhere, pass `--data-dir /absolute/path/to/data`. This directory must contain both feature files for the selected dataset.

## SR-CPPO and baselines

Run the commands below from this directory. These examples use five seeds; `--seeds` specifies one or more independent runs. Defaults are seed 42, 1,500 training episodes, 200 steps per episode, and 200 frozen-policy evaluation episodes.

```bash
python run.py --method sr-cppo --dataset optc --seeds 42 43 45 46 47
python run.py --method sr-cppo --dataset dapt2020 --seeds 42 43 45 46 47

python run.py --method ppo --dataset optc --seeds 42 43 45 46 47
python run.py --method cpo --dataset optc --seeds 42 43 45 46 47
python run.py --method cppo-pid --dataset optc --seeds 42 43 45 46 47
python run.py --method cvar-ppo --dataset optc --seeds 42 43 45 46 47
python run.py --method rule-based --dataset optc --seeds 42 43 45 46 47
```

The baselines also support `--dataset dapt2020`. Rule-Based performs frozen evaluation directly and does not use `--train-eps`. Use `--train-eps` and `--eval-eps` to change episode counts. For all available options, run `python run.py --help`.

The default output directory is `outputs/<dataset>/<method>/`. It contains logs, `run_config.json`, per-seed results, summary CSV files, and the method's checkpoints. Use `--output` to specify a different directory. Separate dataset and method directories keep identically named checkpoints from different experiments apart. For repeated runs of the same configuration, specify a new `--output` directory.

SR-CPPO enables deployment fallback by default. The `--no-fallback` option disables fallback only during frozen evaluation. To evaluate existing Best-Safe weights:

```bash
python run.py --method sr-cppo --dataset optc --seeds 46 --eval-only --output /path/to/checkpoint_directory
```

The directory must contain `optc_v721_best_safe_seed46.pt`. SR-CPPO retains this checkpoint naming convention for compatibility with existing weights, including runs on DAPT 2020. The output directory and `run_config.json` identify the dataset. This option performs frozen evaluation; it does not resume training.

Example background execution over SSH on Linux:

```bash
mkdir -p logs
nohup python -u run.py --method cpo --dataset optc --seeds 42 43 45 46 47 > logs/cpo_optc.out 2>&1 < /dev/null &
```

## OpTC ablations

The ablation entry point provides separate configurations, step-level traces, checkpoint role validation, and training resume support. The default seeds are `42 43 45 46 47`, with the following default variants:

| Variant | Component change |
|---|---|
| `no_decayed_cvar` | Disable temporal decay weighting |
| `no_anti_windup` | Disable the anti-windup mechanism specified by this variant |
| `no_fallback` | Disable fallback during training and deployment |
| `no_best_safe` | Disable Best-Safe selection; optionally reuse the full model's Final checkpoint |
| `no_criticality` | Use equal weights and the corresponding gating settings |

```bash
python run_ablation.py --seeds 42 43 45 46 47 --output outputs/ablation
```

```bash
python run_ablation.py --variants no_best_safe --seeds 46 --full-manifest full_checkpoints.json --output outputs/ablation
python run_ablation.py --variants no_criticality --seeds 46 --resume --output outputs/ablation
```

Ablations use the `indexed` evaluation protocol by default; `legacy` remains available. Run `python run_ablation.py --help` for all options. The ablation code targets OpTC and does not include a DAPT ablation configuration.

## Method references

- PPO: Schulman et al. (2017). *Proximal Policy Optimization Algorithms*.
- CPO: Achiam et al. (2017). *Constrained Policy Optimization*. ICML.
- CPPO-PID: Stooke et al. (2020). *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods*. ICML.
- CVaR-PPO: Ying et al. (2022). *Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk*. This implementation uses an upper-tail episodic-cost constraint. Static CVaR follows the formulation of Rockafellar and Uryasev (2000).
- Rule-Based: A static threshold response heuristic for this task. Background: Stakhanova et al. (2007), *A taxonomy of intrusion response systems*.
