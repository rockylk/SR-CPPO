# SR-CPPO

[English](README.md) · [旧版代码存档](https://github.com/rockylk/SR-CPPO/tree/archive/pre-update-20261116)

SR-CPPO、五个对比方法及 OpTC 组件消融的模块化实现。主方法通过命令行参数切换 OpTC / DAPT 2020，两套数据共用训练和评估代码。

## 目录

```text
SR-CPPO/
├── run.py                         # 主方法和基线入口
├── run_ablation.py                # OpTC 消融入口
├── requirements.txt
├── data/
│   ├── optc/                      # OpTC 特征数组
│   └── dapt2020/                  # DAPT 2020 特征与可选标签
└── cyberdefense/
    ├── common/                    # 数据校验、路径、日志
    ├── methods/
    │   ├── sr_cppo/
    │   ├── ppo/
    │   ├── cpo/
    │   ├── cppo_pid/
    │   ├── cvar_ppo/
    │   └── rule_based/
    └── ablation/                  # 组件变体、训练、评估、断点恢复
```

各学习方法按 `models.py`、`environment.py`、`risk.py`、`agent.py`、`training.py`、`evaluation.py`、`runner.py` 拆分。`runtime.py` 负责随机种子与日志。Rule-Based 不训练网络，因此没有训练模块。消融的 `io.py` 管理检查点，`base_agent.py` 保留其使用的 SR-CPPO 核心。

## 环境

建议使用 Python 3.10–3.12。激活已有 PyTorch 环境后，在本目录执行：

```bash
python -m pip install -r requirements.txt
```

已有 GPU 版 PyTorch 时可直接使用，入口默认在 CUDA 可用时使用 GPU。`--device cpu` 使用 CPU；`--threads 1` 可限制 CPU 线程。运行结果会记录 Python、NumPy、PyTorch 版本及设备。

## 数据

| 数据集 | 良性特征文件 | 攻击特征文件 | 维度 |
|---|---|---|---|
| OpTC | `data/optc/optc_ot_sequence_benign2.npy` | `data/optc/optc_ot_sequence_malicious2.npy` | 24,988 × 16 / 3,338 × 16 |
| DAPT 2020 | `data/dapt2020/dapt_ot_sequence_benign2.npy` | `data/dapt2020/dapt_ot_sequence_malicious2.npy` | 17,018 × 16 / 1,026 × 16 |

训练输入均为现有提取流程输出的 16 维数组，不在此入口重新提取或转换。DAPT 目录中的 `dapt_ot_labels_*2.npy` 是配套标签，训练环境读取上述特征文件。

默认数据路径相对于项目目录解析，与终端当前目录无关。若数据另存于其他位置，用 `--data-dir /absolute/path/to/data`，该目录内需同时包含所选数据集的两份特征文件。

## 主方法与基线

在本目录运行。以下以五个种子为例；`--seeds` 可明确指定单个或多个独立运行。默认单种子 42，训练 1,500 回合、每回合 200 步，冻结评估 200 回合。

```bash
python run.py --method sr-cppo --dataset optc --seeds 42 43 45 46 47
python run.py --method sr-cppo --dataset dapt2020 --seeds 42 43 45 46 47

python run.py --method ppo --dataset optc --seeds 42 43 45 46 47
python run.py --method cpo --dataset optc --seeds 42 43 45 46 47
python run.py --method cppo-pid --dataset optc --seeds 42 43 45 46 47
python run.py --method cvar-ppo --dataset optc --seeds 42 43 45 46 47
python run.py --method rule-based --dataset optc --seeds 42 43 45 46 47
```

基线同样通过 `--dataset dapt2020` 切换数据。Rule-Based 直接执行冻结评估；`--train-eps` 不用于该方法。可通过 `--train-eps`、`--eval-eps` 修改回合数，查看完整参数用 `python run.py --help`。

默认输出为 `outputs/<dataset>/<method>/`，含日志、`run_config.json`、逐种子结果、汇总 CSV 以及算法原有的检查点。可用 `--output` 显式指定目录。数据集和方法分目录存放，因此不同实验的同名检查点不会相互覆盖。重复运行同一组合时，请指定新的 `--output`。

SR-CPPO 的部署 fallback 默认开启；`--no-fallback` 仅关闭冻结评估时的 fallback。只评估已有 Best-Safe 权重：

```bash
python run.py --method sr-cppo --dataset optc --seeds 46 --eval-only --output /path/to/checkpoint_directory
```

该目录需包含 `optc_v721_best_safe_seed46.pt`。为兼容已有权重，SR-CPPO 检查点沿用此命名，包括 DAPT 2020 运行；数据集由输出目录和 `run_config.json` 标识。该选项是冻结评估，不是训练断点续跑。

SSH 后台运行示例（Linux）：

```bash
mkdir -p logs
nohup python -u run.py --method cpo --dataset optc --seeds 42 43 45 46 47 > logs/cpo_optc.out 2>&1 < /dev/null &
```

## OpTC 消融

消融入口保留独立的配置、逐步轨迹、检查点角色校验及断点恢复。默认种子为 `42 43 45 46 47`，默认变体如下：

| 参数名 | 组件变化 |
|---|---|
| `no_decayed_cvar` | 关闭时间衰减权重 |
| `no_anti_windup` | 关闭该变体指定的 anti-windup 机制 |
| `no_fallback` | 训练及部署关闭 fallback |
| `no_best_safe` | 关闭 Best-Safe 选择；可复用完整模型的 Final 检查点 |
| `no_criticality` | 使用等权配置及其对应门控设置 |

```bash
python run_ablation.py --seeds 42 43 45 46 47 --output outputs/ablation
```

`no_best_safe` 在提供完整模型 Final 权重时直接复用；未提供时，按该变体训练并关闭 Best-Safe 选择。`full_reference` 只评估已有 Best-Safe 权重，缺少权重时记录为不可用。可用 `--full-checkpoint-dir` 指定目录，或用 `--full-manifest` 指定 JSON：

```json
{
  "checkpoints": {
    "46": {
      "best_safe": "checkpoints/optc_v721_best_safe_seed46.pt",
      "final": "checkpoints/optc_v721_final_seed46.pt"
    }
  }
}
```

路径相对于 JSON 所在目录；Final 权重需包含 `role: "final"` 或 `"full_final"`，且权重内的种子必须匹配。Best-Safe 不能代替 Final。更多种子按相同结构添加。

```bash
python run_ablation.py --variants no_best_safe --seeds 46 --full-manifest full_checkpoints.json --output outputs/ablation
python run_ablation.py --variants no_criticality --seeds 46 --resume --output outputs/ablation
```

消融默认使用 `indexed` 评估协议，保留 `legacy` 选项。完整参数见 `python run_ablation.py --help`。消融代码针对 OpTC，未增加 DAPT 消融配置。

## 方法文献

- PPO: Schulman et al. (2017). *Proximal Policy Optimization Algorithms*.
- CPO: Achiam et al. (2017). *Constrained Policy Optimization*. ICML.
- CPPO-PID: Stooke et al. (2020). *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods*. ICML.
- CVaR-PPO: Ying et al. (2022). *Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk*；本实现使用上尾回合成本约束。静态 CVaR 采用 Rockafellar and Uryasev (2000) 的表述。
- Rule-Based: 本任务的静态阈值响应启发式；背景文献为 Stakhanova et al. (2007), *A taxonomy of intrusion response systems*。
