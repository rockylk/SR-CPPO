"""OpTC component ablations of SR-CPPO.
Policy optimization and checkpoint selection."""

from pathlib import Path

from .base_agent import SR_CPPO_v721
from .config import SCHEMA
from .io import atomic_torch_save, snapshot_models
from .risk import AblationPID
from .runtime import logger


class AblationAgent(SR_CPPO_v721):

    def __init__(self, arm, run_dir, prefix, metadata, seed=42, device="cpu"):
        gate_high_val = 0.3 if not arm.criticality_aware else 0.7
        super().__init__(obs_dim=16, gate_high=gate_high_val, seed=seed, device=device)
        self.arm = arm
        self.run_dir = Path(run_dir)
        self.prefix = prefix
        self.metadata = metadata
        self.pid = AblationPID(
            enabled=arm.anti_windup,
            kp=0.5,
            ki=0.02,
            kd=0.3,
            lam_max=30.0,
            leak=0.985,
            rate_limit=2.0,
            dead_band_frac=0.1,
        )
        self.safe_ckpt_path = str(self.run_dir / f"{prefix}_best_safe.pt")
        self.snapshot_ep = -1

    def _bucket_cvars(self):
        if self.arm.temporal_memory:
            return super()._bucket_cvars()
        return tuple(
            (
                self._cvar_with_decay(values, decay=1.0)
                for values in (self.cost_buf_benign, self.cost_buf_mal)
            )
        )

    def _apply_sanity_gate(self, enabled=True):
        return super()._apply_sanity_gate(enabled=enabled and self.arm.anti_windup)

    def risk_fallback_active(self):
        return self.arm.fallback and super().risk_fallback_active()

    def _save_safe_checkpoint(self, ep, metrics):
        self.best_safe_state = snapshot_models(self)
        self.best_safe_metrics = dict(metrics)
        checkpoint = self._state_dict_for_ckpt()
        checkpoint.update(
            schema=SCHEMA,
            role="best_safe",
            metadata=self.metadata,
            safe_metrics=dict(metrics),
            saved_episode=ep,
        )
        atomic_torch_save(checkpoint, self.safe_ckpt_path)
        logger.info(
            "[%s] Best-Safe saved at ep=%d score=%.4f", self.prefix, ep, self.best_safe_score
        )
