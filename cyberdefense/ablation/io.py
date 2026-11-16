"""OpTC component ablations of SR-CPPO.
Atomic checkpoints, metadata and result serialization."""

import os
import csv
import random
import numpy as np
import torch
from collections import deque
import hashlib
import json
import time
from pathlib import Path

from .config import SCHEMA


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, deque)):
        return [jsonable(v) for v in x]
    return x


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_object(x):
    return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    replace_file(temp, path)


def replace_file(source, destination):
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    replace_file(temp, path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8-sig", newline="") as f:
            fields = next(csv.reader(f))
        if fields != list(row):
            raise ValueError(f"CSV schema mismatch: {path}")
    with path.open("a", newline="", encoding="utf-8-sig" if not exists else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def snapshot_models(agent):
    return {
        short: {k: v.detach().clone() for k, v in getattr(agent, long).state_dict().items()}
        for short, long in (
            ("enc", "encoder"),
            ("act", "actor"),
            ("cr", "critic_r"),
            ("cc", "critic_c"),
        )
    }


def restore_models(agent, states):
    for short, long in (
        ("enc", "encoder"),
        ("act", "actor"),
        ("cr", "critic_r"),
        ("cc", "critic_c"),
    ):
        getattr(agent, long).load_state_dict(states[short])


def safe_tree(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, deque):
        return {"__deque__": True, "maxlen": x.maxlen, "items": [safe_tree(v) for v in x]}
    if isinstance(x, dict):
        return {k: safe_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [safe_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple((safe_tree(v) for v in x))
    return x


def restore_tree(x):
    if isinstance(x, dict) and x.get("__deque__"):
        return deque(x["items"], maxlen=x["maxlen"])
    if isinstance(x, dict):
        return {k: restore_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [restore_tree(v) for v in x]
    return x


def atomic_torch_save(obj, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(safe_tree(obj), temporary)
    replace_file(temporary, path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_resume(agent, path, next_episode, counters):
    excluded = {"encoder", "actor", "critic_r", "critic_c", "opt", "pid", "cost_norm", "arm"}
    attrs = {k: v for k, v in vars(agent).items() if k not in excluded}
    np_state = np.random.get_state()
    checkpoint = {
        "schema": SCHEMA,
        "role": "resume",
        "metadata": agent.metadata,
        "models": snapshot_models(agent),
        "optimizer": agent.opt.state_dict(),
        "attributes": attrs,
        "pid": vars(agent.pid),
        "cost_norm": vars(agent.cost_norm),
        "next_episode": next_episode,
        "counters": counters,
        "rng": {
            "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    atomic_torch_save(checkpoint, path)


def resume_agent(agent, path):
    checkpoint = load_torch(path)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("role") != "resume":
        raise ValueError("Not a complete ablation resume checkpoint")
    restore_models(agent, checkpoint["models"])
    agent.opt.load_state_dict(checkpoint["optimizer"])
    for k, v in restore_tree(checkpoint["attributes"]).items():
        if k not in {"run_dir", "safe_ckpt_path", "metadata", "device"}:
            setattr(agent, k, v)
    vars(agent.pid).update(restore_tree(checkpoint["pid"]))
    vars(agent.cost_norm).update(restore_tree(checkpoint["cost_norm"]))
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    state = rng["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return (int(checkpoint["next_episode"]), checkpoint["counters"])


def checkpoint_models(checkpoint):
    if "models" in checkpoint:
        return checkpoint["models"]
    long_names = ("encoder", "actor", "critic_r", "critic_c")
    if all((k in checkpoint for k in long_names)):
        return {a: checkpoint[b] for a, b in zip(("enc", "act", "cr", "cc"), long_names)}
    if all((k in checkpoint for k in ("enc", "act", "cr", "cc"))):
        return {k: checkpoint[k] for k in ("enc", "act", "cr", "cc")}
    raise ValueError("Checkpoint must contain encoder, actor and both critics")


def load_weights(agent, path, expected_role=None, external=False):
    checkpoint = load_torch(path)
    seed = checkpoint.get("seed", checkpoint.get("metadata", {}).get("seed"))
    if seed is None or int(seed) != agent.seed:
        raise ValueError(
            f"Checkpoint seed mismatch or missing seed: {path}; stored={seed}, expected={agent.seed}"
        )
    if external:
        declared_variant = checkpoint.get("metadata", {}).get("variant")
        if declared_variant and declared_variant not in ("full", "full_reference"):
            raise ValueError(
                "An ablated variant checkpoint cannot substitute for the original Full trajectory"
            )
    if expected_role == "final":
        role = checkpoint.get("role")
        if role not in ("final", "full_final"):
            raise ValueError(
                f"Final role not verified in {path}; Best-Safe cannot substitute for Final"
            )
    restore_models(agent, checkpoint_models(checkpoint))
    agent.best_safe_ep = int(checkpoint.get("best_safe_ep", -1))
    agent.best_safe_metrics = checkpoint.get("safe_metrics", {})
    return checkpoint


def select_training_checkpoint(agent):
    if agent.best_safe_state is not None:
        restore_models(agent, agent.best_safe_state)
        return "best_safe"
    if agent.best_state is not None:
        restore_models(agent, agent.best_state)
        return "best_snapshot_fallback"
    return "final_weights_fallback"
