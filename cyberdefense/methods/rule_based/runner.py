"""Static threshold response heuristic. Background: Stakhanova et al. (2007), A taxonomy of intrusion response systems.
Experiment orchestration and result export."""

import os
import csv
import numpy as np

from .agent import StaticRuleBasedAgent
from cyberdefense.common.data import data_paths
from .environment import OpTC_DataDrivenEnv
from .evaluation import evaluate_frozen_baseline
from .runtime import logger, set_global_seed


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
    t_crit_map = {2: 12.7062, 3: 4.3027, 4: 3.1824, 5: 2.7764, 10: 2.2622}
    t_crit = t_crit_map.get(n, 1.96)
    logger.info("\n" + "=" * 95)
    logger.info(
        f"[{model_tag}] FINAL 5-SEED STATISTICAL SUMMARY (Seeds: {list(seeds)}, Eval Eps/Seed: {eval_eps})"
    )
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
        mean_sd_str = f"{mean:.2f} ± {sd:.2f}"
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
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
    per_seed_csv = os.path.join(os.getcwd(), f"{model_tag.lower()}_5seeds_raw.csv")
    summary_csv = os.path.join(os.getcwd(), f"{model_tag.lower()}_5seeds_summary.csv")
    with open(per_seed_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_seed_results)
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"[Saved] {model_tag} Per-seed raw CSV -> {per_seed_csv}")
    logger.info(f"[Saved] {model_tag} 5-Seed Summary CSV -> {summary_csv}")
    return (all_seed_results, summary_rows)


def run_multi_seed_experiments(seeds=(42, 43, 44, 45, 46), total_eps=1500, eval_eps=200):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    benign_path, mal_path = resolve_data_paths(current_dir)
    env = OpTC_DataDrivenEnv(benign_path, mal_path, logger=logger)
    agent = StaticRuleBasedAgent(env.benign_data, theta_low=0.4, theta_high=0.7)
    all_results = []
    for s in seeds:
        set_global_seed(s)
        logger.info(f"\n[Rule-Based Heuristic | Seed={s}] Evaluating over {eval_eps} episodes...")
        res = evaluate_frozen_baseline(env, agent, eval_eps=eval_eps)
        res["seed"] = s
        all_results.append(res)
    return summarize_and_save_multi_seed(all_results, "Rule_Based", seeds, eval_eps, logger)
