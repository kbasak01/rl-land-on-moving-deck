"""SAC CPU-vs-GPU throughput, measured once before tuning (P5-D9; rl-trainer rule).

Trains the committed configs/rl/sac.yaml for a short budget on each device into scratch run
directories (never artifacts/runs/), train pool only, and prints steps/s after the random
warm-up. Nothing here is a tuning trial: no score is read.

Usage: python scripts/p5_sac_device_bench.py --out-root <scratch dir>
"""

import argparse
import json
import time
from pathlib import Path

from rld.rl.config import RL_CONFIG_DIR, apply_overrides, load_yaml, train_config_from_dict
from rld.rl.train import train


def main() -> None:
    """Run the benchmark on cpu and cuda and print one JSON line per device."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()
    raw = load_yaml(RL_CONFIG_DIR / "sac.yaml")
    for device in ("cpu", "cuda"):
        over = {
            "method": f"sac_bench_{device}",
            "run_group": f"sac_bench_{device}",
            "device": device,
            "total_steps": args.steps,
            "sac.learning_starts": 5000,
            "eval.interval_steps": 10**9,
            "checkpoint_interval_steps": 10**9,
        }
        cfg = train_config_from_dict(apply_overrides(raw, over), f"bench {device}")
        run_dir = args.out_root / device
        run_dir.mkdir(parents=True, exist_ok=False)
        t0 = time.perf_counter()
        status = train(cfg, 0, run_dir)
        wall = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "device": device,
                    "steps": status["steps"],
                    "wall_s": round(wall, 1),
                    "fps": status.get("fps"),
                    "fps_excl_eval": status.get("fps_excl_eval"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
