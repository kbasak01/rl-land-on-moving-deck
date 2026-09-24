"""Scratch: same held-command noise signal, attitude/PID loop at 30 Hz vs 240 Hz. Train pool."""

import dataclasses
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def run(args):
    ctrl_hz, std, descend = args
    from rld.eval.envs import load_eval_configs, make_env, motion_for

    cfgs = load_eval_configs()
    cfgs = dataclasses.replace(
        cfgs, landing=dataclasses.replace(cfgs.landing, ctrl_freq_hz=ctrl_hz)
    )
    hold = ctrl_hz // 30
    out, env = [], None
    for i, (hdg, seed) in enumerate(itertools.product([180.0, 135.0], range(8))):
        m = motion_for(cfgs, "frigate", "SS3", hdg, 6.0, seed)
        env = make_env(cfgs, m, "frigate", "aft", 0) if env is None else env
        env.set_motion(m)
        for rep in range(3):
            env.reset(seed=1000 * i + rep)
            rng = np.random.default_rng(1000 * i + rep)
            k, done = 0, False
            while not done:
                if k % hold == 0:  # a new command every 1/30 s whatever the loop rate
                    a = rng.normal(0, std, 3) + np.array([0, 0, -descend])
                _, _, term, trunc, _ = env.step(a.astype(np.float32))
                done, k = term or trunc, k + 1
            out.append(env.record.as_row()["outcome"])
    env.close()
    v, c = np.unique(out, return_counts=True)
    return ctrl_hz, std, descend, dict(zip(v.tolist(), (c / len(out)).round(2).tolist(), strict=True))


if __name__ == "__main__":
    grid = [(hz, s, d) for hz in (30, 240) for s in (0.2, 0.37) for d in (0.0, 0.2)] + [
        (240, 0.05, 0.2),
        (30, 0.05, 0.2),
    ]
    with ProcessPoolExecutor(10) as ex:
        for hz, s, d, r in ex.map(run, grid):
            print(f"ctrl={hz:>3}Hz std={s:<5} descend={d:<4} n=48 {r}", flush=True)
