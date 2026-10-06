# Third-party notices

This project is MIT-licensed (see [LICENSE](LICENSE)). It depends on two projects included as git
submodules, contains a few short passages derived from them, and depends on others at runtime.
They are listed here.

## Git submodules (`third_party/`, not modified)

Both submodules are pinned commits, installed editable, and never edited (CLAUDE.md rule 7). Their
own license files travel with them.

### gym-pybullet-drones

- **Upstream:** <https://github.com/learnsyslab/gym-pybullet-drones>, pinned at
  `7ebad1ecabd28a7000add2d05f888aa2e837c2cc` (`v1.0.0-324-g7ebad1e`).
- **License:** MIT, Copyright (c) 2020 Jacopo Panerati.
- **What is used:** at runtime, the `BaseAviary` environment and the `DSLPIDControl` velocity
  tracker, plus the Crazyflie 2.x model (`cf2x.urdf`, `cf2.dae`) from its `assets/` directory. The
  landing GIFs in `results/figures/gifs/` are renders of that model. They were rendered from the
  submodule; the model files are not copied into this repository.
- **What is derived in this repository:** `DeckLandingAviary._housekeeping`
  (`src/rld/envs/landing_env.py`, marked in the source) reproduces upstream
  `BaseAviary._housekeeping`, unchanged except for the flags it loads the drone with (P5-D5). The
  MIT notice above covers it. `tests/test_lazy_trajectory.py` pins the upstream version it copies.

### deck-motion-forecast (`dmf`)

- **Upstream:** <https://github.com/kbasak01/deck-motion-forecast>, pinned at
  `e9fa15cc35312a5c1ecf0c42668a690b5061c090`.
- **License:** MIT, Copyright (c) 2026 K. Basak.
- **What is used:** deck motion (`dmf.sim.response.synthesize_motion`, evaluated analytically on
  the physics grid), its vessel and corpus configs, its forecasters, and its deployment harness
  (`dmf.deploy`). It is consumed read-only, **including its known roll/pitch–heave phase
  defect**, which this project carries and controls for rather than fixes.
- **What is derived in this repository:**
  - `configs/deck/mss_s175_ss5.yaml` is a copy of dmf's `configs/mss/s175_ss5.yaml` with one
    path changed. Its header says so, and `tests/test_deck_mss.py` asserts it.
  - `src/rld/deploy/parity.py::_result` reproduces dmf's parity arithmetic, marked in the source.
  - `src/rld/deploy/latency.py`'s `benchmark_ort` and `benchmark_torch` mirror dmf's
    `dmf.deploy.bench` functions line for line, with only the input-rank check changed (marked in
    the module docstring).
  - `src/rld/deck/bridge.py` reproduces one line of dmf's realization seed path, marked in the
    source.

## Fetched at build time, not redistributed

### Marine Systems Simulator (MSS)

- **Upstream:** <https://github.com/cybergalactic/MSS>, pinned at
  `98970f71a21cfe81e7e29abdcc1bb6741789cddc` (`Makefile` `MSS_SHA`).
- **License:** MIT, Copyright (c) 2004 Thor I. Fossen.
- **Use:** only the optional MSS transfer arm of Phase 7 uses it. `make mss-export` clones it into
  the gitignored `artifacts/mss/upstream/`, and dmf's patched MSS m-file is staged there verbatim
  (SHA-checked). No MSS file is committed to this repository. dmf's own
  `THIRD_PARTY_NOTICES.md` documents its modified MSS file.

## Runtime dependencies

Python dependencies are pinned in [`pyproject.toml`](pyproject.toml). Each carries its own
license, and none is vendored here. The licenses below are as declared in the installed packages'
metadata:

| package | version | license |
|---|---|---|
| PyBullet | 3.2.7 | zlib |
| Stable-Baselines3 | 2.9.0 | MIT |
| Gymnasium | 1.3.0 | MIT |
| PyTorch | 2.13.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT (as declared; PyTorch's own license is BSD-3-Clause-style) |
| ONNX | 1.22.0 | Apache-2.0 |
| ONNX Runtime (GPU) | 1.29.0 | MIT |
| TensorRT (`tensorrt-cu13`) | 10.16.1.11 | NVIDIA proprietary license (Phase 8 latency study only) |
| NumPy | 2.5.2 | BSD-3-Clause (with bundled components) |
| SciPy | 1.18.1 | BSD-3-Clause |
| pandas | 3.0.5 | BSD-3-Clause |
| PyArrow | 25.0.1 | Apache-2.0 |
| Matplotlib | 3.11.1 | Matplotlib license (PSF-based) |
| imageio | 2.37.4 | BSD-2-Clause |
| Pillow | 12.3.0 | MIT-CMU |
| PyYAML | 6.0.3 | MIT |
| rliable | 1.2.0 | Apache-2.0 (pinned; its stratified bootstrap is re-implemented in `rld.eval.stats`, P3-D1 §4) |

The GIF captions are set in DejaVu Sans, read from Matplotlib's bundled copy. DejaVu fonts carry
the Bitstream Vera / DejaVu free license.

## Published methods

PPO, SAC, residual policy learning, the interquartile mean and stratified bootstrap, the Wilson
interval, the JONSWAP spectrum and Froude scaling are published methods. They are cited in the
README's [Citations](README.md#citations) section. No text or code from those papers is
reproduced in this repository.
