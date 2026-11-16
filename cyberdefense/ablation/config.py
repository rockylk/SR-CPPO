"""OpTC component ablations of SR-CPPO.
Default run settings. Command-line arguments select the dataset and run budget."""

from dataclasses import dataclass

RUN_SINGLE_SEED_QUICK_TEST = True

SINGLE_SEED = 42

MULTI_SEEDS = (42, 43, 44, 45, 46)

TOTAL_TRAIN_EPS = 1500

FROZEN_EVAL_EPS = 200

ENABLE_A1_FALLBACK_IN_EVAL = True

EVAL_ONLY_FROM_SAVED_CKPT = False

SEEDS = (42, 43, 45, 46, 47)

SCHEMA = "optc-ablation-v1"

IMPLEMENTATION_VERSION = "1.0"

DATASET = "OpTC"

REFERENCE_WEIGHTS = (0.2, 0.9)

METRICS = (
    "Avg_Reward",
    "R_J",
    "Vmin_A",
    "FDR_pct",
    "AIR_pct",
    "CVaR_B",
    "CVaR_M",
    "CVaR_mix",
    "Vmin_B",
)

CORE_HASH = "93a028b039919b1330fcc3010d6e81935f90f2d88fe677ee645712409705f294"

TRAIN_ARMS = ("no_decayed_cvar", "no_anti_windup", "no_fallback", "no_best_safe", "no_criticality")

LABELS = {
    "full_reference": "Full SR-CPPO (existing checkpoint)",
    "no_decayed_cvar": "w/o Decayed-CVaR",
    "no_anti_windup": "w/o PID anti-windup",
    "no_fallback": "w/o Safe-Fallback (train + deployment)",
    "no_best_safe": "w/o Best-Safe (Full final checkpoint)",
    "no_criticality": "w/o criticality weighting",
    "no_decayed_no_anti_windup": "w/o Decayed-CVaR and anti-windup",
    "deployment_no_fallback": "Full checkpoint w/o deployment fallback",
}


@dataclass(frozen=True)
class Arm:
    name: str

    @property
    def temporal_memory(self):
        return self.name not in ("no_decayed_cvar", "no_decayed_no_anti_windup")

    @property
    def anti_windup(self):
        return self.name not in ("no_anti_windup", "no_decayed_no_anti_windup")

    @property
    def fallback(self):
        return self.name not in ("no_fallback", "deployment_no_fallback")

    @property
    def best_safe(self):
        return self.name != "no_best_safe"

    @property
    def criticality_aware(self):
        return self.name != "no_criticality"

    @property
    def weights(self):
        return (0.55, 0.55) if self.name == "no_criticality" else REFERENCE_WEIGHTS


def condition_tag(args):
    return (
        "nominal"
        if args.confidence_cap is None
        else "cap" + str(args.confidence_cap).replace(".", "p")
    )
