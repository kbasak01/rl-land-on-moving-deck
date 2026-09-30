"""Scratch: cross the 200k/final weights and VecNormalize stats; SS3 tune draw, serial."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def run(pair):
    wk, sk = pair
    from rld.control.tuning import draw_tuning_episodes, load_tuning
    from rld.eval.envs import load_eval_configs, make_env, motion_for
    from rld.rl.train import load_policy

    run_dir = "artifacts/runs/ppo_smoke_p5d7/0"
    cfgs = load_eval_configs()
    eps = [e for e in draw_tuning_episodes(cfgs.sim, load_tuning()) if e.ss == "SS3"]
    pol = load_policy(run_dir, wk)
    pol.normalizer = load_policy(run_dir, sk).normalizer
    env, out = None, []
    for e in eps:
        m = motion_for(cfgs, e.vessel, e.ss, e.heading_deg, e.speed_kn, e.realization_seed)
        env = make_env(cfgs, m, e.vessel, e.pad, 0) if env is None else env
        env.set_motion(m)
        obs, _ = env.reset(seed=e.episode_seed)
        pol.reset(e.episode_seed)
        done = False
        while not done:
            obs, _, t, tr, _ = env.step(pol.act(obs))
            done = t or tr
        out.append(env.record.as_row()["outcome"])
    return wk, sk, dict(Counter(out))


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        for w, s, r in ex.map(run, [("final", 200000), (200000, "final")]):
            print(f"weights={w} stats={s} {r}", flush=True)
