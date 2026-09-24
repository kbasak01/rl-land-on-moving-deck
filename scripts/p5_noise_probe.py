"""Scratch probe: outcome of zero-mean-noise policies vs std and temporal smoothness.

Train-pool realizations only (frigate SS3, 180/135 deg, seeds 0-7). Not a trial; nothing is
written to artifacts/ or results/.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def run(args):
    std, rho, descend = args
    from rld.eval.envs import load_eval_configs, make_env, motion_for

    cfgs = load_eval_configs()
    out = []
    env = None
    for i, (hdg, seed) in enumerate(itertools.product([180.0, 135.0], range(8))):
        m = motion_for(cfgs, "frigate", "SS3", hdg, 6.0, seed)
        if env is None:
            env = make_env(cfgs, m, "frigate", "aft", 0)
        else:
            env.set_motion(m)
        for rep in range(3):
            env.reset(seed=1000 * i + rep)
            rng = np.random.default_rng(1000 * i + rep)
            n = np.zeros(3)
            done = False
            while not done:
                # AR(1) noise with stationary std `std`; rho=0 is i.i.d. (PPO's default)
                n = rho * n + np.sqrt(1 - rho**2) * rng.normal(0, std, 3)
                a = n + np.array([0.0, 0.0, -descend])
                _, _, term, trunc, info = env.step(a.astype(np.float32))
                done = term or trunc
            out.append(env.record.as_row()["outcome"])
    env.close()
    vals, cnt = np.unique(out, return_counts=True)
    return std, rho, descend, dict(zip(vals.tolist(), (cnt / len(out)).round(2).tolist(), strict=True))


if __name__ == "__main__":
    grid = [(s, r, d) for s in (0.05, 0.1, 0.2, 0.37) for r in (0.0, 0.9) for d in (0.0, 0.2)]
    with ProcessPoolExecutor(16) as ex:
        for std, rho, d, res in ex.map(run, grid):
            print(f"std={std:<5} rho={rho:<4} descend={d:<4} n=48 {res}", flush=True)
