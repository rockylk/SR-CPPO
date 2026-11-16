"""SR-CPPO recurrent policy, distributional cost critic and PID controller.
Experiment orchestration and result export."""

import os
import csv
import numpy as np

from cyberdefense.common.data import data_paths
from .environment import OpTC_DataDrivenEnv
from .runtime import logger
from .training import train_and_eval_single_seed_v721


def run_experiments(
    seeds=(42,), total_eps=1500, eval_eps=200, enable_fallback_in_eval=True, eval_only=False
):
    benign_path, mal_path = data_paths()
    env = OpTC_DataDrivenEnv(
        benign_path,
        mal_path,
        horizon=200,
        attack_rate=0.55,
        V_threshold=0.6,
        V_min_target=0.65,
        idle_bonus=0.03,
    )
    all_seed_results = []
    for s in seeds:
        res = train_and_eval_single_seed_v721(
            env,
            seed=s,
            total_eps=total_eps,
            eval_eps=eval_eps,
            enable_fallback_in_eval=enable_fallback_in_eval,
            eval_only=eval_only,
        )
        all_seed_results.append(res)
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
        ("R_J_at_BestSafe_Ckpt", "R_J Decayed at Best-Safe Ep (Ref)"),
    ]
    n = len(all_seed_results)
    t_crit_map = {1: 0.0, 2: 12.7062, 3: 4.3027, 4: 3.1824, 5: 2.7764, 10: 2.2622}
    t_crit = t_crit_map.get(n, 1.96)
    fb_tag = "with_fallback" if enable_fallback_in_eval else "wo_fallback"
    mode_title = (
        "SINGLE-SEED QUICK VERIFICATION" if n == 1 else f"FINAL {n}-SEED STATISTICAL SUMMARY"
    )
    logger.info("\n" + "=" * 95)
    logger.info(
        f"{mode_title} (Seeds: {list(seeds)}, Eval Episodes/Seed: {eval_eps}, Mode: {fb_tag})"
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
        mean_sd_str = f"{mean:.2f} ± {sd:.2f}" if n > 1 else f"{mean:.2f} (Single Seed)"
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if n > 1 else f"[{mean:.2f}, {mean:.2f}]"
        logger.info(f"{label:<45} | {mean_sd_str:<22} | {ci_str:<22}")
        summary_rows.append(
            {
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
    suffix = f"seed{seeds[0]}_{fb_tag}" if n == 1 else f"{n}seeds_{fb_tag}"
    per_seed_csv = os.path.join(os.getcwd(), f"sr_cppo_v721_{suffix}_raw.csv")
    summary_csv = os.path.join(os.getcwd(), f"sr_cppo_v721_{suffix}_summary.csv")
    with open(per_seed_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_seed_results)
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"[Saved] Per-seed raw results -> {per_seed_csv}")
    logger.info(f"[Saved] Summary table        -> {summary_csv}")
    return (all_seed_results, summary_rows)
