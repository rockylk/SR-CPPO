"""PPO. Reference: Schulman et al. (2017), Proximal Policy Optimization Algorithms.
Experiment orchestration and result export."""

import os
import csv
import numpy as np

from cyberdefense.common.data import data_paths
from .environment import OpTC_DataDrivenEnv
from .runtime import logger
from .training import train_single_seed_ppo


def resolve_data_paths(current_dir=None):
    return data_paths()


def summarize_and_save_multi_seed(all_seed_results, model_tag, seeds, eval_eps, logger):
    metric_keys = [
        ("Avg_Reward", "Avg Reward"),
        ("R_J", "R_J Stratified Tail Risk"),
        ("Vmin_A", "Vmin_A Attack Survivability"),
        ("FDR_pct", "FDR False Defense Rate %"),
        ("AIR_pct", "AIR Attack Intercept Rate %"),
        ("CVaR_B", "CVaR_B Benign Tail Risk"),
        ("CVaR_M", "CVaR_M Malicious Tail Risk"),
        ("CVaR_mix", "CVaR_mix Pooled Mixture CVaR"),
        ("Vmin_B", "Vmin_B Benign Survivability (Extended)"),
    ]
    n = len(all_seed_results)
    t_crit_map = {1: 0.0, 2: 12.7062, 3: 4.3027, 4: 3.1824, 5: 2.7764, 10: 2.2622}
    t_crit = t_crit_map.get(n, 1.96)
    title_str = (
        "SINGLE-SEED QUICK VERIFICATION" if n == 1 else f"FINAL {n}-SEED STATISTICAL SUMMARY"
    )
    logger.info("\n" + "=" * 95)
    logger.info(f"[{model_tag}] {title_str} (Seeds: {list(seeds)}, Eval Eps/Seed: {eval_eps})")
    logger.info("=" * 95)
    logger.info(f"{'Metric':<45} | {'Mean ± SD':<22} | {'95% CI':<22}")
    logger.info("-" * 95)
    summary_rows = []
    for key, label in metric_keys:
        vals = np.array([r[key] for r in all_seed_results], dtype=np.float64)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        half_ci = float(t_crit * sd / np.sqrt(n)) if n > 1 else 0.0
        ci_low = mean - half_ci
        ci_high = mean + half_ci
        mean_sd_str = f"{mean:.2f} ± {sd:.2f}" if n > 1 else f"{mean:.2f} (Single Seed)"
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if n > 1 else f"[{mean:.2f}, {mean:.2f}]"
        logger.info(f"{label:<45} | {mean_sd_str:<22} | {ci_str:<22}")
        summary_rows.append(
            {
                "Model": model_tag,
                "Metric_Key": key,
                "Description": label,
                "Mean": round(mean, 4),
                "SD": round(sd, 4),
                "CI95_HalfWidth": round(half_ci, 4),
                "CI95_Low": round(ci_low, 4),
                "CI95_High": round(ci_high, 4),
                "Formatted_Mean_SD": mean_sd_str,
                "Formatted_95CI": ci_str,
            }
        )
    logger.info("=" * 95)
    suffix = f"seed{seeds[0]}" if n == 1 else f"{n}seeds"
    per_seed_csv = os.path.join(os.getcwd(), f"{model_tag.lower()}_{suffix}_raw.csv")
    summary_csv = os.path.join(os.getcwd(), f"{model_tag.lower()}_{suffix}_summary.csv")
    with open(per_seed_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_seed_results)
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"[Saved] {model_tag} Per-seed raw CSV -> {per_seed_csv}")
    logger.info(f"[Saved] {model_tag} Summary CSV      -> {summary_csv}")
    return (all_seed_results, summary_rows)


def run_multi_seed_experiments(seeds=(42,), total_eps=1500, eval_eps=200):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    benign_path, mal_path = resolve_data_paths(current_dir)
    env = OpTC_DataDrivenEnv(benign_path, mal_path, logger=logger)
    all_results = []
    for s in seeds:
        res = train_single_seed_ppo(env, seed=s, total_eps=total_eps, eval_eps=eval_eps)
        all_results.append(res)
    return summarize_and_save_multi_seed(all_results, "PPO", seeds, eval_eps, logger)
