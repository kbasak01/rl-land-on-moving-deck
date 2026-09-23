"""Fit dmf's forecasters on this project's development pool, driving dmf as a library.

Phase 4, protocol P4-D1/P4-D2. ``third_party/deck-motion-forecast`` is read-only, and dmf's
own experiment driver selects training data by **regime name only**, so it cannot be told to
fit on the P3-D2 pool. This module therefore calls dmf's public building blocks directly, in
the same order dmf's ``train/experiment.py`` does, and writes every output under
``artifacts/dmf/`` (gitignored) plus one committed record, ``results/forecast/``.

The pool, and why it is not dmf's ``id``/train
----------------------------------------------
dmf's ``id`` train partition holds SS6 and 90 deg realizations, which are frozen
``unseen_seastate`` and ``unseen_heading`` **test** episodes. Fitting on it would leak.
Every model here is fitted on ``rld.deck.splits.dev_pool(sim_cfg)[0]`` (729 realizations:
frigate, SS3-SS5, 45/135/180 deg, 0/6/12 kn, seed ordinals 0-26), and ``dev_pool()[1]`` (135
realizations, seed ordinals 27-31) is the dmf ``val`` partition: early stopping, seed
selection and the residual-interval calibration. The dmf :class:`~dmf.data.splits.Split` is
built with ``regime="dev_pool"`` -- a label only, which dmf never branches on -- so that the
saved :class:`~dmf.data.normalize.NormStats` and dmf's own fit reports carry
``fitted_on="dev_pool/train"`` / ``"dev_pool/val"`` rather than a misleading ``"id/..."``.
``test_keys`` is empty: nothing here is scored.

The four models (P4-D2)
-----------------------
============================  ===============  =====================================
artifact name                 fit              dmf configs
============================  ===============  =====================================
``dlinear_ols``               closed form      model/dlinear_ols.yaml, e02 train block
``residual_interval``         closed form      model/residual_interval.yaml, e03
``tcn``                       SGD, 3 seeds     model/tcn.yaml, e02_deep.yaml train
``tcn_quantile``              SGD, 3 seeds     model/tcn_quantile.yaml, e03 train
============================  ===============  =====================================

The closed-form rows share **one** training-moments pass. ``residual_interval``'s point half
*is* ``dlinear_ols``; its 9-level residual fan is fitted on the tune pool by dmf's
:func:`~dmf.train.closed_form.fit_residual_interval`, which refuses any partition but ``val``.

**Seed selection rule, pre-registered in the plan:** the deployed seed of an SGD model is the
one with the lowest tune-pool loss in its own objective (MSE for ``tcn``, pinball for
``tcn_quantile``), ties to the lower seed. Selection is recomputed over every seed record on
disk, so seeds fitted in separate invocations are compared on equal terms.

Outputs per model, ``<out>/<model>/``
-------------------------------------
``state_dict.pt`` (the deployed weights -- dmf never saves ``dlinear_ols``),
``norm_stats.npz`` (the train-partition per-channel scale dmf normalises by; dmf does not
store it in checkpoints), ``model.onnx`` (``dmf.deploy.export_onnx.export_model``, quantile
sort node included), ``parity.json`` (``dmf.deploy.parity.check_parity``, worst of the draws),
``meta.json`` (everything :class:`rld.deck.forecast.OnlineForecaster` needs to rebuild the
model, plus the files' SHA-256s), and for SGD models ``seeds/`` (every seed's checkpoint and
loss curves). Interval models also get ``conformal_padvz.npz`` (below).

Pad-v_z conformal calibration (plan amendment 2026-09-23)
----------------------------------------------------------
For each interval model and each centreline pad of ``configs/deck/pad.yaml`` (aft, cg),
:func:`calibrate_pad_vz` fits one multiplicative factor per lead (150) that rescales the
interval-arithmetic pad-``v_z`` box about its point to 90 % coverage on the tune pool. The
conventions are dmf's split-conformal ones (``dmf.eval.conformal``): the width-normalised CQR
score ``s = max((m - y)/(m - lo), (y - m)/(hi - m))``, ``gamma`` the
``ceil((n+1)(1 - alpha))``-th order statistic (never an interpolated quantile), half-widths
floored at ``MIN_HALF_WIDTH`` with the floored count reported, and the calibration windows a
fixed stride over the tune partition's window index capped at ``CONFORMAL_MAX_WINDOWS`` --
the same windows dmf's residual-interval fit draws. dmf's realization-level sensitivity (the
same order statistic over per-realization median scores) is stored beside it, never used.
The lever arm is the frigate's (the tune pool is frigate only). ``--calibrate-only`` runs the
step on an existing artifact directory, e.g. after the TCN fits.

Units
-----
Everything a model sees or emits is dmf's corpus, **full scale** at 10 Hz: roll and pitch in
degrees, their rates in degrees per second, heave in metres, heave rate in metres per second.
Lookback and horizons are in samples (200 = 20 s full, 150 = 15 s full). Losses are
dimensionless (normalised space). Wall-clock times are seconds. Nothing in this module is
model scale.
"""

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from dmf.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SimConfig,
    TrainConfig,
    load_data,
    load_experiment,
    load_model,
    load_sim,
)
from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import NormStats
from dmf.data.splits import RealizationKey, Regime, Split, load_manifest, realization_key
from dmf.data.windows import WindowSpec, window_spec_from_config, window_start_indices
from dmf.deploy.export_onnx import export_model
from dmf.deploy.parity import (
    PARITY_N_WINDOWS,
    PARITY_SEED,
    PARITY_SEEDS,
    ParityResult,
    check_parity,
    random_parity_windows,
)
from dmf.eval.conformal import CONFORMAL_MAX_WINDOWS, MIN_HALF_WIDTH, order_statistic_index
from dmf.models.base import BaseForecaster
from dmf.sim.generate import realization_grid, realization_path
from dmf.train.closed_form import (
    TrainingMoments,
    accumulate_training_moments,
    fit_dlinear_ols,
    fit_residual_interval,
)
from dmf.train.loop import fit, set_seed, validate
from dmf.train.losses import resolve_loss
from dmf.train.registry import MODEL_REGISTRY, build_model

from rld.config import REPO_ROOT
from rld.deck.bridge import FORECAST_CHANNELS, load_vessel_cached
from rld.deck.config import MOTION_JONSWAP_CONFIG, PAD_CONFIG, load_motion, load_pads
from rld.deck.splits import dev_pool, key_for_spec, spec_for_key
from rld.provenance import environment_provenance

__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_CORPUS_ROOT",
    "DEFAULT_RECORD_DIR",
    "DMF_ROOT",
    "FITTED_ON",
    "FORECASTERS",
    "POOL_LABEL",
    "SMOKE",
    "ForecasterSpec",
    "SmokeSettings",
    "INTERVAL_MODELS",
    "PRIMARY_ROLE",
    "SECONDARY_ROLE",
    "calibrate_only",
    "calibrate_pad_vz",
    "canonical_key_list",
    "check_corpus",
    "corpus_frame",
    "export_seed",
    "seed_export_name",
    "fit_forecasters",
    "key_list_sha256",
    "load_norm_scale",
    "pool_keys",
    "select_seed",
    "sha256_file",
]

#: The read-only dmf submodule. Its configs are read in place and never copied.
DMF_ROOT: Path = REPO_ROOT / "third_party" / "deck-motion-forecast"

#: dmf's configs directory.
DMF_CONFIGS: Path = DMF_ROOT / "configs"

#: dmf's corpus-generation config -- the one ``configs/deck/motion_jonswap.yaml`` points at.
DMF_CORPUS_CONFIG: Path = DMF_CONFIGS / "sim" / "corpus.yaml"

#: dmf's task definition: 10 Hz, lookback 200, horizons up to 150 samples, six channels.
DMF_DATA_CONFIG: Path = DMF_CONFIGS / "data" / "default.yaml"

#: Where ``make dmf-forecasters`` writes the corpus. Full scale, 2304 realizations, ~1.1 GB.
DEFAULT_CORPUS_ROOT: Path = REPO_ROOT / "artifacts" / "dmf" / "corpus"

#: Root of the per-model artifact directories.
DEFAULT_ARTIFACT_ROOT: Path = REPO_ROOT / "artifacts" / "dmf"

#: The committed record: ``fit_manifest.json`` and ``fit_keys.json``.
DEFAULT_RECORD_DIR: Path = REPO_ROOT / "results" / "forecast"

#: Label of the dmf ``Split``. dmf uses it for provenance strings only.
POOL_LABEL: str = "dev_pool"

#: Provenance of every fitted coefficient and of the normalisation scale.
FITTED_ON: str = f"{POOL_LABEL}/train"

#: Provenance of early stopping, seed selection and the residual quantiles.
TUNED_ON: str = f"{POOL_LABEL}/tune"

#: Schema version of ``fit_manifest.json`` and ``meta.json``.
SCHEMA_VERSION: int = 1

#: The rule the deployed seed is chosen by, written verbatim into the manifest.
SELECTION_RULE: str = (
    "lowest tune-pool loss in the model's own training objective (mse for point heads, "
    "pinball for quantile heads); ties to the lower seed"
)

FitKind = Literal["closed_form", "sgd"]

#: The models whose pad-v_z band is conformal-calibrated (quantile heads).
INTERVAL_MODELS: tuple[str, ...] = ("residual_interval", "tcn_quantile")

#: ``role`` of a model directory holding the pre-registered, deployed model.
PRIMARY_ROLE: str = "primary (pre-registered selection)"

#: ``role`` of a secondary seed export (results-skeptic M5): reported, never selected.
SECONDARY_ROLE: str = "seed-sensitivity, not selected"

#: Nominal miscoverage of the pad-v_z calibration (the 90 % band).
CONFORMAL_ALPHA: float = 0.1

#: Vessel the calibration's lever arms are taken from: the only hull in the tune pool.
CALIBRATION_VESSEL: str = "frigate"


@dataclass(frozen=True)
class ForecasterSpec:
    """One forecaster this project fits, and the dmf configs that define it.

    Attributes:
        name: Artifact directory name, equal to the dmf config's ``label``.
        model_config: dmf model config, read in place.
        experiment_config: dmf experiment config whose ``train`` block (epochs, batch,
            learning rate, patience, AMP, workers) is used unchanged.
        kind: ``closed_form`` (solved from moments, deterministic) or ``sgd`` (seeded).
    """

    name: str
    model_config: Path
    experiment_config: Path
    kind: FitKind


#: The P4-D2 model set, in fit order. Closed-form rows first: they take seconds and share
#: one moments pass.
FORECASTERS: dict[str, ForecasterSpec] = {
    spec.name: spec
    for spec in (
        ForecasterSpec(
            "dlinear_ols",
            DMF_CONFIGS / "model" / "dlinear_ols.yaml",
            DMF_CONFIGS / "experiment" / "e02_deep.yaml",
            "closed_form",
        ),
        ForecasterSpec(
            "residual_interval",
            DMF_CONFIGS / "model" / "residual_interval.yaml",
            DMF_CONFIGS / "experiment" / "e03_probabilistic.yaml",
            "closed_form",
        ),
        ForecasterSpec(
            "tcn",
            DMF_CONFIGS / "model" / "tcn.yaml",
            DMF_CONFIGS / "experiment" / "e02_deep.yaml",
            "sgd",
        ),
        ForecasterSpec(
            "tcn_quantile",
            DMF_CONFIGS / "model" / "tcn_quantile.yaml",
            DMF_CONFIGS / "experiment" / "e03_probabilistic.yaml",
            "sgd",
        ),
    )
}


@dataclass(frozen=True)
class SmokeSettings:
    """A pipeline check at toy size: every code path, none of the compute.

    Attributes:
        n_train_keys: Training realizations kept, spread over the sorted pool.
        n_tune_keys: Tune realizations kept.
        epochs: SGD epoch cap.
        parity_windows: Random windows per parity draw.
        parity_seeds: Parity draws.
    """

    n_train_keys: int = 6
    n_tune_keys: int = 3
    epochs: int = 1
    parity_windows: int = 50
    parity_seeds: tuple[int, ...] = (PARITY_SEED,)


#: The default smoke settings used by ``--smoke``.
SMOKE: SmokeSettings = SmokeSettings()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key_list(keys: Sequence[RealizationKey] | frozenset[RealizationKey]) -> str:
    """Serialise realization keys canonically: sorted, compact JSON.

    Args:
        keys: dmf realization keys ``(ss, heading_deg, speed_kn, vessel, seed)``.

    Returns:
        ``json.dumps`` of the sorted keys as lists, no whitespace. Heading and speed are
        floats (degrees, knots), the seed an integer ordinal.
    """
    ordered = sorted(realization_key(*key) for key in keys)
    return json.dumps([list(key) for key in ordered], separators=(",", ":"))


def key_list_sha256(keys: Sequence[RealizationKey] | frozenset[RealizationKey]) -> str:
    """Return the SHA-256 of :func:`canonical_key_list`, the pool identity the manifest records."""
    return hashlib.sha256(canonical_key_list(keys).encode("utf-8")).hexdigest()


def _spread(keys: list[RealizationKey], n: int) -> list[RealizationKey]:
    """Pick ``n`` keys evenly spread over a sorted list (deterministic, no RNG)."""
    if n >= len(keys):
        return keys
    idx = np.linspace(0, len(keys) - 1, n).round().astype(int)
    return [keys[int(i)] for i in idx]


def pool_keys(
    sim_cfg: SimConfig, smoke: SmokeSettings | None = None
) -> tuple[list[RealizationKey], list[RealizationKey]]:
    """Return the (train, tune) realization keys the forecasters are fitted on.

    Args:
        sim_cfg: dmf's corpus config; the pools are derived from its grid by
            :func:`rld.deck.splits.dev_pool`, never from literal seed numbers.
        smoke: If given, keep only an evenly spread handful of each pool.

    Returns:
        ``(train, tune)``, each sorted. 729 and 135 keys for the committed grid.
    """
    train_specs, tune_specs = dev_pool(sim_cfg)
    train = sorted(key_for_spec(spec) for spec in train_specs)
    tune = sorted(key_for_spec(spec) for spec in tune_specs)
    if smoke is not None:
        train = _spread(train, smoke.n_train_keys)
        tune = _spread(tune, smoke.n_tune_keys)
    return train, tune


def check_corpus(corpus_root: Path, sim_cfg: SimConfig, required: Sequence[RealizationKey]) -> str:
    """Assert that the corpus on disk is the corpus our bridge reproduces.

    Three checks: (1) ``configs/deck/motion_jonswap.yaml`` points at the same dmf corpus
    config this module loads; (2) the manifest's record geometry (``duration_s``,
    ``spinup_s``, ``fs_hz``, ``n_components``, ``jitter``) equals that config's and its key
    set is exactly ``realization_grid(sim_cfg)``; (3) every required key has a file.

    Args:
        corpus_root: The corpus root written by dmf's ``generate_corpus.py``.
        sim_cfg: dmf's corpus config.
        required: Keys the fit will read.

    Returns:
        The SHA-256 of ``manifest.parquet``, recorded as the corpus identity.

    Raises:
        ValueError: On any mismatch.
    """
    motion = load_motion(MOTION_JONSWAP_CONFIG)
    if motion.sim_config_path.resolve() != DMF_CORPUS_CONFIG.resolve():
        raise ValueError(
            f"configs/deck/motion_jonswap.yaml points at {motion.sim_config_path}, but the "
            f"forecasters are fitted on the corpus of {DMF_CORPUS_CONFIG}"
        )
    manifest = load_manifest(corpus_root)
    expected = {
        "duration_s": sim_cfg.duration_s,
        "spinup_s": sim_cfg.spinup_s,
        "fs_hz": sim_cfg.fs_hz,
        "n_components": sim_cfg.n_components,
        "jitter": sim_cfg.jitter_frequencies,
    }
    for column, value in expected.items():
        seen = set(manifest[column].tolist())
        if seen != {value}:
            raise ValueError(f"corpus manifest column {column!r} holds {seen}, expected {value}")
    on_disk = {
        realization_key(*row)
        for row in manifest[["ss", "heading", "speed", "vessel", "seed"]].itertuples(index=False)
    }
    grid = {key_for_spec(spec) for spec in realization_grid(sim_cfg)}
    if on_disk != grid:
        raise ValueError(
            f"corpus manifest holds {len(on_disk)} realizations that are not the "
            f"{len(grid)}-realization grid of {DMF_CORPUS_CONFIG}"
        )
    missing = [key for key in required if key not in on_disk]
    if missing:
        raise ValueError(f"{len(missing)} required realizations absent from the corpus")
    return sha256_file(corpus_root / "manifest.parquet")


def _datasets(
    corpus_root: Path, data_cfg: DataConfig, train: list[RealizationKey], tune: list[RealizationKey]
) -> tuple[DeckMotionDataset, DeckMotionDataset]:
    """Build dmf's train and val datasets over the dev pool.

    The val dataset is built **with the train statistics**, as dmf's driver does.

    Returns:
        ``(train, tune)`` datasets. Windows are full scale at 10 Hz.
    """
    split = Split(
        regime=cast(Regime, POOL_LABEL),
        train_keys=frozenset(train),
        val_keys=frozenset(tune),
        test_keys=frozenset(),
    )
    spec = window_spec_from_config(data_cfg)
    train_ds = DeckMotionDataset(corpus_root, split, "train", data_cfg, spec)
    tune_ds = DeckMotionDataset(
        corpus_root, split, "val", data_cfg, spec, stats=train_ds.norm_stats
    )
    if train_ds.norm_stats.fitted_on != FITTED_ON:
        raise ValueError(f"normalisation fitted on {train_ds.norm_stats.fitted_on!r}")
    return train_ds, tune_ds


def _save_norm_stats(path: Path, stats: NormStats, train_ds: DeckMotionDataset) -> None:
    """Write the train-partition normalisation statistics dmf does not checkpoint.

    ``scale`` is what dmf divides the de-meaned window by (population std over every
    training sample, per channel). ``train_mean`` is recorded for reference only: dmf
    de-means each window by **its own** mean, never by a global one.
    """
    series = train_ds.training_series
    np.savez(
        path,
        channels=np.asarray(stats.channels),
        scale=np.asarray(stats.scale, dtype=np.float64),
        train_mean=series.mean(axis=0),
        fitted_on=np.asarray(stats.fitted_on),
        n_realizations=np.asarray(stats.n_realizations),
    )


def load_norm_scale(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read ``norm_stats.npz`` back.

    Returns:
        ``(channels, scale)``; ``scale`` float64, per channel, in corpus units (degrees,
        degrees per second, metres, metres per second -- full scale).
    """
    with np.load(path, allow_pickle=False) as data:
        channels = tuple(str(c) for c in data["channels"].tolist())
        scale = np.asarray(data["scale"], dtype=np.float64)
    return channels, scale


def _loss_name(cfg: ModelConfig) -> str:
    """Return the name of a model's own objective."""
    return {"point": "mse", "quantile": "pinball", "gaussian": "gaussian_nll"}[cfg.head]


def _tune_loss(
    model: BaseForecaster,
    cfg: ModelConfig,
    tune_ds: DeckMotionDataset,
    train_cfg: TrainConfig,
    num_workers: int,
) -> float:
    """Return a model's tune-pool loss in its own objective, dimensionless (CPU)."""
    loader = make_dataloader(
        tune_ds,
        batch_size=max(train_cfg.batch_size, 1024),
        shuffle=False,
        num_workers=num_workers,
        seed=0,
    )
    model.to("cpu").eval()
    return float(validate(model, loader, train_cfg, loss_fn=resolve_loss(cfg.head, cfg.quantiles)))


def _export_and_check(
    model: BaseForecaster,
    sample: torch.Tensor,
    model_dir: Path,
    smoke: SmokeSettings | None,
) -> dict[str, Any]:
    """Export to ONNX and run dmf's torch-vs-ORT (CPU EP) parity check.

    Returns:
        The parity record: the **worst** draw's :class:`~dmf.deploy.parity.ParityResult`
        fields plus every draw's ``max_abs_err`` and the verdict.

    Raises:
        RuntimeError: If any draw fails dmf's scale-relative criterion.
    """
    model.to("cpu").eval()
    onnx_path = export_model(model, sample, model_dir / "model.onnx")
    seeds = smoke.parity_seeds if smoke is not None else PARITY_SEEDS
    n_windows = smoke.parity_windows if smoke is not None else PARITY_N_WINDOWS
    results: list[ParityResult] = []
    for seed in seeds:
        windows = random_parity_windows(model.lookback, model.n_input_channels, n_windows, seed)
        results.append(check_parity(model, onnx_path, windows, seed=seed))
    worst = max(results, key=lambda r: r.max_abs_err / r.scaled_tolerance)
    record: dict[str, Any] = {
        "criterion": "max_abs_err < tolerance * max(1, |y|_max) (dmf P7-D3), worst draw",
        "passed": all(r.passed for r in results),
        "passed_absolute": all(r.passed_absolute for r in results),
        "worst": {
            key: getattr(worst, key)
            for key in (
                "max_abs_err",
                "mean_abs_err",
                "n_windows",
                "tolerance",
                "passed",
                "output_abs_max",
                "scale",
                "scaled_tolerance",
                "passed_absolute",
                "seed",
                "provider",
            )
        },
        "draws": [{"seed": r.seed, "max_abs_err": r.max_abs_err} for r in results],
    }
    (model_dir / "parity.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not record["passed"]:
        raise RuntimeError(f"torch vs ORT parity failed for {model_dir.name}: {record['worst']}")
    return record


def _model_meta(
    spec: ForecasterSpec,
    cfg: ModelConfig,
    data_cfg: DataConfig,
    window: WindowSpec,
    selected_seed: int,
    smoke: bool,
    files: Mapping[str, str],
    *,
    name: str | None = None,
    role: str = PRIMARY_ROLE,
) -> dict[str, Any]:
    """Return ``meta.json``: how to rebuild the model, and what was fitted on what."""
    return {
        "schema": SCHEMA_VERSION,
        "model": name or spec.name,
        "role": role,
        "parent": spec.name,
        "dmf_name": cfg.name,
        "label": cfg.label,
        "head": cfg.head,
        "quantiles": list(cfg.quantiles),
        "params": cfg.params,
        "kind": spec.kind,
        "lookback": window.lookback,
        "max_horizon": window.max_horizon,
        "fs_hz": data_cfg.fs_hz,
        "input_channels": list(data_cfg.input_channels),
        "target_channels": list(data_cfg.target_dofs),
        "revin": data_cfg.revin,
        "fitted_on": FITTED_ON,
        "tuned_on": TUNED_ON,
        "selected_seed": selected_seed,
        "smoke": smoke,
        "model_config": str(spec.model_config.relative_to(DMF_ROOT)),
        "model_config_sha256": sha256_file(spec.model_config),
        "files": dict(files),
    }


def _write_deployed(
    spec: ForecasterSpec,
    cfg: ModelConfig,
    model: BaseForecaster,
    data_cfg: DataConfig,
    window: WindowSpec,
    train_ds: DeckMotionDataset,
    tune_ds: DeckMotionDataset,
    model_dir: Path,
    selected_seed: int,
    smoke: SmokeSettings | None,
    *,
    name: str | None = None,
    role: str = PRIMARY_ROLE,
) -> dict[str, Any]:
    """Write a deployable model directory and return its record for the manifest.

    ``name``/``role`` label a secondary seed-sensitivity export (:func:`export_seed`); the
    default is the pre-registered, selected model.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu").eval()
    torch.save(model.state_dict(), model_dir / "state_dict.pt")
    _save_norm_stats(model_dir / "norm_stats.npz", train_ds.norm_stats, train_ds)
    sample = tune_ds[0][0][None]
    parity = _export_and_check(model, sample, model_dir, smoke)
    files = {
        name: sha256_file(model_dir / name)
        for name in ("state_dict.pt", "norm_stats.npz", "model.onnx")
    }
    meta = _model_meta(
        spec, cfg, data_cfg, window, selected_seed, smoke is not None, files, name=name, role=role
    )
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "state_dict_sha256": files["state_dict.pt"],
        "onnx_sha256": files["model.onnx"],
        "norm_stats_sha256": files["norm_stats.npz"],
        "meta_sha256": sha256_file(model_dir / "meta.json"),
        "parity": parity,
    }


def select_seed(records: Mapping[int, float]) -> int:
    """Apply the pre-registered selection rule: lowest tune loss, ties to the lower seed.

    Args:
        records: ``{seed: tune-pool loss}``, dimensionless.

    Returns:
        The selected seed.

    Raises:
        ValueError: If ``records`` is empty or any loss is not finite.
    """
    if not records:
        raise ValueError("no seed records to select from")
    if not all(math.isfinite(loss) for loss in records.values()):
        raise ValueError(f"non-finite tune loss among {dict(records)}")
    return min(records, key=lambda seed: (records[seed], seed))


def _fit_closed_form(
    spec: ForecasterSpec,
    cfg: ModelConfig,
    moments: TrainingMoments,
    train_ds: DeckMotionDataset,
    tune_ds: DeckMotionDataset,
    train_cfg: TrainConfig,
    num_workers: int,
) -> tuple[BaseForecaster, dict[str, Any]]:
    """Solve one closed-form model and measure its tune loss (CPU, deterministic)."""
    n_in = len(train_ds.input_columns)
    n_out = len(train_ds.target_columns)
    kernel_size = int(cfg.params["kernel_size"])
    ridge = float(cfg.params.get("ridge", 0.0))
    lookback = train_ds.window_spec.lookback
    started = time.perf_counter()
    report: dict[str, Any]
    model: BaseForecaster
    if spec.name == "residual_interval":
        interval, interval_report = fit_residual_interval(
            moments,
            tune_ds,
            kernel_size=kernel_size,
            ridge=ridge,
            lookback=lookback,
            n_input_channels=n_in,
            n_target_channels=n_out,
            n_quantiles=len(cfg.quantiles),
            num_workers=num_workers,
        )
        model = interval
        report = {
            "fitted_on_dmf_label": interval_report.fitted_on,
            "n_windows_total": interval_report.n_windows_total,
            "n_windows_used": interval_report.n_windows_used,
            "window_stride": interval_report.window_stride,
            "residual_val_mse": interval_report.residual_val_mse,
            "point_residual_train_mse": interval_report.point.residual_train_mse,
            "point_cond_r": interval_report.point.cond_r,
            "n_fitted_parameters": interval_report.n_fitted_parameters,
            "tune_loss_is_in_sample": True,
        }
    else:
        ols, ols_report = fit_dlinear_ols(
            moments,
            kernel_size=kernel_size,
            ridge=ridge,
            lookback=lookback,
            n_input_channels=n_in,
            n_target_channels=n_out,
        )
        model = ols
        report = {
            "residual_train_mse": ols_report.residual_train_mse,
            "cond_r": ols_report.cond_r,
            "n_fitted_parameters": ols_report.n_fitted_parameters,
            "tune_loss_is_in_sample": False,
        }
    report["solve_time_s"] = time.perf_counter() - started
    tune_loss = _tune_loss(model, cfg, tune_ds, train_cfg, num_workers)
    seed_record = {
        "seed": 0,
        "deterministic": True,
        "tune_loss": tune_loss,
        "tune_loss_name": _loss_name(cfg),
        "fit_report": report,
    }
    return model, seed_record


def _fit_sgd_seed(
    cfg: ModelConfig,
    train_ds: DeckMotionDataset,
    tune_ds: DeckMotionDataset,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    seed: int,
    device: str,
    seeds_dir: Path,
) -> dict[str, Any]:
    """Train one SGD model at one seed with dmf's loop and write its seed record.

    Fresh loaders per seed, shuffled with generator seed ``seed``, so one seed's run is a
    function of its seed alone, whichever other seeds share the invocation. (dmf's own
    driver builds the loaders once and lets the shuffle generator run on across seeds; that
    makes seed 1 depend on whether seed 0 ran first in the same process.)
    """
    spec = train_ds.window_spec
    train_loader = make_dataloader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        seed=seed,
    )
    tune_loader = make_dataloader(
        tune_ds,
        batch_size=max(train_cfg.batch_size, 1024),
        shuffle=False,
        num_workers=train_cfg.num_workers,
        seed=0,
    )
    set_seed(seed)
    model = build_model(
        cfg, spec, len(train_ds.input_columns), len(train_ds.target_columns), revin=data_cfg.revin
    ).to(device)
    result = fit(
        model,
        train_loader,
        tune_loader,
        train_cfg,
        seed,
        seeds_dir,
        loss_fn=resolve_loss(cfg.head, cfg.quantiles),
        label=cfg.label,
    )
    record = {
        "seed": seed,
        "deterministic": False,
        "tune_loss": float(result.best_val_loss),
        "tune_loss_name": _loss_name(cfg),
        "tune_loss_amp": train_cfg.amp_dtype if device.startswith("cuda") else "off",
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "epoch_cap": train_cfg.epochs,
        "wall_time_s": result.wall_time_s,
        "device": device,
        "train_losses": list(result.train_losses),
        "val_losses": list(result.val_losses),
        "checkpoint": result.checkpoint_path.name,
        "checkpoint_sha256": sha256_file(result.checkpoint_path),
    }
    (seeds_dir / f"seed{seed}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _seed_records(seeds_dir: Path) -> dict[int, dict[str, Any]]:
    """Read every seed record on disk for one SGD model."""
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(seeds_dir.glob("seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[int(record["seed"])] = record
    return records


def _config_hashes(specs: Sequence[ForecasterSpec]) -> dict[str, str]:
    """Return ``{dmf-relative path: SHA-256}`` for every dmf config the fits read."""
    paths = {DMF_CORPUS_CONFIG, DMF_DATA_CONFIG}
    for spec in specs:
        paths.update({spec.model_config, spec.experiment_config})
    return {str(p.relative_to(DMF_ROOT)): sha256_file(p) for p in sorted(paths)}


def _merge_manifest(path: Path, header: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """Merge this invocation's models into an existing manifest, refusing a pool change.

    Raises:
        ValueError: If the existing manifest was written for different pools, a different
            dmf commit or a different corpus -- two fits on different data must not share a
            record.
    """
    manifest: dict[str, Any] = {**header, "models": {}}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for field in ("pools", "dmf_sha", "corpus", "smoke"):
            if previous.get(field) != header[field]:
                raise ValueError(
                    f"{path} was written with a different {field!r} "
                    f"({previous.get(field)!r} vs {header[field]!r}); delete it and refit "
                    f"every model rather than mix two fits in one record"
                )
        manifest["models"] = dict(previous.get("models", {}))
        manifest["configs"] = {**previous.get("configs", {}), **header["configs"]}
    manifest["models"].update(models)
    manifest["models"] = {name: manifest["models"][name] for name in sorted(manifest["models"])}
    return manifest


def fit_forecasters(
    *,
    models: Sequence[str],
    seeds: Sequence[int] | None,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    record_dir: Path = DEFAULT_RECORD_DIR,
    device: str = "cuda",
    num_workers: int | None = None,
    smoke: SmokeSettings | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Fit the requested forecasters on the dev pool and write artifacts plus the record.

    Args:
        models: Names from :data:`FORECASTERS`.
        seeds: SGD seeds to fit in this invocation, or None for the experiment config's
            (0, 1, 2). Closed-form models ignore it.
        corpus_root: dmf corpus root (full scale, 10 Hz).
        out_root: Artifact root; each model writes ``out_root/<model>/``.
        record_dir: Where ``fit_manifest.json`` and ``fit_keys.json`` are written.
        device: Torch device for SGD training. Closed-form solves, tune losses of
            closed-form models, export and parity always run on CPU.
        num_workers: DataLoader workers, overriding the dmf train block's 16.
        smoke: Toy-size pipeline check; never writes to the committed record unless
            ``record_dir`` is pointed there explicitly.
        log: Progress sink.

    Returns:
        The manifest as written.

    Raises:
        ValueError: On an unknown model, a corpus mismatch, or a manifest conflict.
        RuntimeError: If a torch-vs-ORT parity check fails.
    """
    unknown = [m for m in models if m not in FORECASTERS]
    if unknown:
        raise ValueError(f"unknown forecaster(s) {unknown}, expected {list(FORECASTERS)}")
    specs = [FORECASTERS[m] for m in FORECASTERS if m in models]
    sim_cfg = load_sim(DMF_CORPUS_CONFIG)
    data_cfg = load_data(DMF_DATA_CONFIG)
    if data_cfg.fs_hz != sim_cfg.fs_hz:
        raise ValueError(f"data fs {data_cfg.fs_hz} Hz != corpus fs {sim_cfg.fs_hz} Hz")
    window = window_spec_from_config(data_cfg)
    train_keys, tune_keys = pool_keys(sim_cfg, smoke)
    if set(train_keys) & set(tune_keys):
        raise ValueError("train and tune pools overlap")
    corpus_sha = check_corpus(corpus_root, sim_cfg, train_keys + tune_keys)
    log(f"[fit] corpus ok ({corpus_root}); pools: {len(train_keys)} train, {len(tune_keys)} tune")

    started = time.perf_counter()
    train_ds, tune_ds = _datasets(corpus_root, data_cfg, train_keys, tune_keys)
    log(
        f"[fit] datasets: {len(train_ds)} train windows, {len(tune_ds)} tune windows "
        f"({time.perf_counter() - started:.1f} s); scale {train_ds.norm_stats.scale.round(4)}"
    )

    moments: TrainingMoments | None = None
    written: dict[str, Any] = {}
    for spec in specs:
        cfg = load_model(spec.model_config)
        experiment: ExperimentConfig = load_experiment(spec.experiment_config)
        kind = MODEL_REGISTRY[cfg.name].FIT_KIND
        if kind != spec.kind:
            raise ValueError(f"{spec.name}: dmf declares FIT_KIND {kind!r}, expected {spec.kind}")
        train_cfg = experiment.train
        if num_workers is not None:
            train_cfg = replace(train_cfg, num_workers=num_workers)
        if smoke is not None:
            train_cfg = replace(train_cfg, epochs=smoke.epochs)
        model_dir = out_root / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        if spec.kind == "closed_form":
            if moments is None:
                t0 = time.perf_counter()
                moments = accumulate_training_moments(
                    train_ds,
                    max_order=1,
                    num_workers=train_cfg.num_workers,
                    decompose_kernel=int(cfg.params["kernel_size"]),
                )
                log(
                    f"[fit] moments over {moments.n_windows} windows "
                    f"({time.perf_counter() - t0:.1f} s)"
                )
            model, record = _fit_closed_form(
                spec, cfg, moments, train_ds, tune_ds, train_cfg, train_cfg.num_workers
            )
            seed_records = {0: record}
            (model_dir / "fit_record.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            selected = 0
        else:
            seeds_dir = model_dir / "seeds"
            seeds_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds if seeds is not None else experiment.seeds:
                log(f"[fit] {spec.name} seed {seed}: up to {train_cfg.epochs} epochs on {device}")
                record = _fit_sgd_seed(
                    cfg, train_ds, tune_ds, train_cfg, data_cfg, int(seed), device, seeds_dir
                )
                log(
                    f"[fit] {spec.name} seed {seed}: tune {record['tune_loss_name']} "
                    f"{record['tune_loss']:.6f} at epoch {record['best_epoch']} of "
                    f"{record['epochs_run']} ({record['wall_time_s']:.0f} s)"
                )
            seed_records = _seed_records(seeds_dir)
            selected = select_seed({s: float(r["tune_loss"]) for s, r in seed_records.items()})
            model = build_model(
                cfg, window, len(train_ds.input_columns), len(train_ds.target_columns)
            )
            state = torch.load(
                seeds_dir / seed_records[selected]["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)

        deployed = _write_deployed(
            spec, cfg, model, data_cfg, window, train_ds, tune_ds, model_dir, selected, smoke
        )
        if spec.name in INTERVAL_MODELS:
            deployed["conformal_padvz"] = calibrate_pad_vz(
                model_dir, corpus_root, window, tune_keys, log=log
            )
            deployed["meta_sha256"] = sha256_file(model_dir / "meta.json")
        expected_seeds = [0] if spec.kind == "closed_form" else list(experiment.seeds)
        written[spec.name] = {
            "kind": spec.kind,
            "dmf_name": cfg.name,
            "head": cfg.head,
            "model_config": str(spec.model_config.relative_to(DMF_ROOT)),
            "train_block_from": str(spec.experiment_config.relative_to(DMF_ROOT)),
            "train_block": {
                "epochs": train_cfg.epochs,
                "batch_size": train_cfg.batch_size,
                "lr": train_cfg.lr,
                "weight_decay": train_cfg.weight_decay,
                "warmup_frac": train_cfg.warmup_frac,
                "grad_clip": train_cfg.grad_clip,
                "patience": train_cfg.patience,
                "amp_dtype": train_cfg.amp_dtype,
                "num_workers": train_cfg.num_workers,
            },
            "fitted_on": FITTED_ON,
            "tuned_on": TUNED_ON,
            "selection_rule": SELECTION_RULE if spec.kind == "sgd" else "deterministic",
            "expected_seeds": expected_seeds,
            "complete": sorted(seed_records) == sorted(expected_seeds),
            "seeds": [
                {k: v for k, v in r.items() if k not in ("train_losses", "val_losses")}
                for _, r in sorted(seed_records.items())
            ],
            "selected_seed": selected,
            "selected_tune_loss": float(seed_records[selected]["tune_loss"]),
            **deployed,
        }
        log(
            f"[fit] {spec.name}: selected seed {selected}, tune "
            f"{_loss_name(cfg)} {written[spec.name]['selected_tune_loss']:.6f}; parity "
            f"max_abs_err {deployed['parity']['worst']['max_abs_err']:.2e}"
        )

    record_dir.mkdir(parents=True, exist_ok=True)
    keys_path = record_dir / "fit_keys.json"
    keys_doc = {
        "about": (
            "Realization keys (ss, heading_deg, speed_kn, vessel, seed_ordinal) the dmf "
            "forecasters were fitted on (train) and early-stopped / seed-selected / "
            "interval-calibrated on (tune). Sorted; see fit_manifest.json for the hashes."
        ),
        "train": json.loads(canonical_key_list(train_keys)),
        "tune": json.loads(canonical_key_list(tune_keys)),
    }
    keys_path.write_text(json.dumps(keys_doc, indent=1) + "\n", encoding="utf-8")
    env = environment_provenance(REPO_ROOT, {"onnxruntime": _ort_version()})
    header: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "about": (
            "Phase 4 forecaster fits (protocol P4-D1, P4-D2). dmf driven as a read-only "
            "library by rld.deck.forecast_fit; artifacts under artifacts/dmf/ (gitignored). "
            "Simulation only."
        ),
        "smoke": smoke is not None,
        "dmf_sha": env["dmf_sha"],
        "corpus": {
            "config": str(DMF_CORPUS_CONFIG.relative_to(DMF_ROOT)),
            "manifest_sha256": corpus_sha,
            "n_realizations": len(realization_grid(sim_cfg)),
        },
        "pools": {
            "fitted_on": FITTED_ON,
            "tuned_on": TUNED_ON,
            "source": "rld.deck.splits.dev_pool(sim_cfg) (protocol P3-D2)",
            "train": {"n": len(train_keys), "sha256": key_list_sha256(train_keys)},
            "tune": {"n": len(tune_keys), "sha256": key_list_sha256(tune_keys)},
            "key_hash": "sha256 of rld.deck.forecast_fit.canonical_key_list(keys)",
            "keys_file": keys_path.name,
        },
        "keys_file_sha256": sha256_file(keys_path),
        "configs": _config_hashes(specs),
        "window": {
            "fs_full_hz": data_cfg.fs_hz,
            "lookback_samples": window.lookback,
            "max_horizon_samples": window.max_horizon,
            "stride_samples": window.stride,
            "channels": list(data_cfg.input_channels),
        },
        "environment": env,
    }
    manifest_path = record_dir / "fit_manifest.json"
    manifest = _merge_manifest(manifest_path, header, written)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"[fit] wrote {manifest_path} and {keys_path}")
    return manifest


#: Leads, seconds full scale, whose factors are echoed into the manifest for reading.
_LEADS_S: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.3, 10.0, 15.0)


def _conformal_gamma(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Return dmf's split-conformal factor per column: the ``k``-th order statistic."""
    k = order_statistic_index(int(scores.shape[0]), alpha)
    picked: np.ndarray = np.partition(scores, k - 1, axis=0)[k - 1]
    return picked


def calibrate_pad_vz(
    model_dir: Path,
    corpus_root: Path,
    window: WindowSpec,
    tune_keys: Sequence[RealizationKey],
    *,
    alpha: float = CONFORMAL_ALPHA,
    threads: int = 8,
    log: Any = print,
) -> dict[str, Any]:
    """Fit and save the per-pad, per-lead pad-v_z conformal factors of one interval model.

    Runs the **deployed** path (ONNX Runtime, CPU) on the tune pool's calibration windows,
    builds the interval-arithmetic box for each centreline pad, scores the true pad ``v_z``
    against it (dmf's CQR ratio), and takes dmf's finite-sample order statistic per lead.
    Writes ``conformal_padvz.npz``, adds its SHA-256 to ``meta.json``, and returns the
    manifest record. Everything is full scale; the factors are dimensionless (and invariant
    to the Froude velocity factor).

    Args:
        model_dir: An interval model's artifact directory (``meta.json`` present).
        corpus_root: dmf corpus root.
        window: The task geometry (lookback 200, horizon 150, stride 5 samples).
        tune_keys: The calibration pool (``dev_pool`` tune).
        alpha: Nominal miscoverage.
        threads: ORT intra-op threads for this offline pass.
        log: Progress sink.

    Returns:
        The record written into the manifest's model entry.

    Raises:
        ValueError: If the model is not an interval model or a factor is not finite.
    """
    from rld.deck.forecast import CONFORMAL_FILE, HI, LO, POINT, OnlineForecaster
    from rld.deck.forecast import pad_vertical_band as box_of

    started = time.perf_counter()
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"].pop(CONFORMAL_FILE, None)  # a stale calibration is not verified, it is redone
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    forecaster = OnlineForecaster(model_dir, "ort", intra_op_threads=threads)
    if forecaster.head != "quantile":
        raise ValueError(f"{model_dir.name} is not an interval model")
    length_full_m = float(load_vessel_cached(CALIBRATION_VESSEL).length_m)
    pads = {
        spec.name: spec.r_pad_frac[0]
        for spec in load_pads(PAD_CONFIG).pads
        if spec.r_pad_frac[1] == 0.0 and spec.r_pad_frac[2] == 0.0
    }
    keys = sorted(tune_keys)
    frames = {key: corpus_frame(corpus_root, key, FORECAST_CHANNELS) for key in keys}
    n_samples = len(next(iter(frames.values())))
    starts = window_start_indices(n_samples, window)
    n_total = len(keys) * starts.size
    stride = max(1, -(-n_total // CONFORMAL_MAX_WINDOWS))  # dmf's calibration stride
    leads = np.arange(1, window.max_horizon + 1)
    scores: dict[str, list[np.ndarray]] = {pad: [] for pad in pads}
    floored: dict[str, int] = dict.fromkeys(pads, 0)
    realization: list[np.ndarray] = []
    for r, key in enumerate(keys):
        index = np.arange(r * starts.size, (r + 1) * starts.size)
        chosen = starts[(index[index % stride == 0]) - r * starts.size]
        if chosen.size == 0:
            continue
        data = frames[key][list(FORECAST_CHANNELS)].to_numpy(np.float64)
        band = forecaster.predict_windows(np.stack([data[s : s + window.lookback] for s in chosen]))
        truth = np.stack([data[s + window.lookback - 1 + leads] for s in chosen])
        realization.append(np.full(chosen.size, r))
        for pad, frac in pads.items():
            x = frac * length_full_m
            _, vz = box_of(band, x)
            y = truth[..., 5] + x * np.cos(np.radians(truth[..., 1])) * np.radians(truth[..., 4])
            lo_raw, hi_raw = vz[..., POINT] - vz[..., LO], vz[..., HI] - vz[..., POINT]
            floored[pad] += int(np.sum((lo_raw < MIN_HALF_WIDTH) | (hi_raw < MIN_HALF_WIDTH)))
            lo_h, hi_h = np.maximum(lo_raw, MIN_HALF_WIDTH), np.maximum(hi_raw, MIN_HALF_WIDTH)
            scores[pad].append(np.maximum((vz[..., POINT] - y) / lo_h, (y - vz[..., POINT]) / hi_h))
    real_index = np.concatenate(realization)
    arrays: dict[str, np.ndarray] = {
        "pads": np.asarray(list(pads)),
        "leads_full_s": leads / 10.0,
        "alpha": np.asarray(alpha),
        "n_windows": np.asarray(real_index.size),
        "window_stride": np.asarray(stride),
        "calibration_vessel": np.asarray(CALIBRATION_VESSEL),
    }
    record: dict[str, Any] = {
        "file": CONFORMAL_FILE,
        "method": (
            "split conformal, dmf.eval.conformal conventions: CQR ratio score on the "
            "interval-arithmetic pad-v_z box, gamma = ceil((n+1)(1-alpha))-th order statistic "
            "per lead, band = point +- gamma * box half-width"
        ),
        "alpha": alpha,
        "calibrated_on": TUNED_ON,
        "n_windows": int(real_index.size),
        "window_stride": stride,
        "n_realizations": len(keys),
        "calibration_vessel": CALIBRATION_VESSEL,
        "pads": {},
    }
    for pad, frac in pads.items():
        s = np.concatenate(scores[pad])
        gamma = _conformal_gamma(s, alpha)
        if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0.0):
            raise ValueError(f"{model_dir.name}/{pad}: non-positive or non-finite gamma")
        medians = np.stack([np.median(s[real_index == r], axis=0) for r in np.unique(real_index)])
        try:
            gamma_real = _conformal_gamma(medians, alpha)
        except ValueError:  # too few realizations (smoke) to certify the level at all
            gamma_real = np.full_like(gamma, np.nan)
        arrays[f"gamma_{pad}"] = gamma
        arrays[f"gamma_realization_{pad}"] = gamma_real
        arrays[f"x_frac_{pad}"] = np.asarray(frac)
        arrays[f"n_floored_{pad}"] = np.asarray(floored[pad])
        at = {f"{lead:.1f}": round(float(gamma[round(lead * 10) - 1]), 6) for lead in _LEADS_S}
        record["pads"][pad] = {
            "x_frac": frac,
            "x_full_m_calibrated": frac * length_full_m,
            "gamma_at_lead_full_s": at,
            "gamma_min": float(gamma.min()),
            "gamma_max": float(gamma.max()),
            "gamma_realization_level_median": float(np.nanmedian(gamma_real)),
            "n_half_widths_floored": floored[pad],
        }
        log(f"[calibrate] {model_dir.name}/{pad}: gamma at {at}")
    path = model_dir / CONFORMAL_FILE
    np.savez(path, **arrays)  # type: ignore[arg-type]
    record["sha256"] = sha256_file(path)
    meta["files"][CONFORMAL_FILE] = record["sha256"]
    meta["conformal_padvz"] = {k: v for k, v in record.items() if k != "pads"}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    record["wall_time_s"] = time.perf_counter() - started
    return record


def calibrate_only(
    *,
    models: Sequence[str],
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    record_dir: Path = DEFAULT_RECORD_DIR,
    smoke: SmokeSettings | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Run :func:`calibrate_pad_vz` on already-fitted interval models and update the record.

    The manifest must already hold each model (it is fitted first); its pool hashes must
    match this invocation's pools.

    Raises:
        ValueError: If a model is not an interval model, is absent from the manifest, or the
            manifest's pools differ.
    """
    bad = [m for m in models if _parent(m) not in INTERVAL_MODELS]
    if bad:
        raise ValueError(f"{bad} are not interval models {INTERVAL_MODELS} or their seed exports")
    sim_cfg = load_sim(DMF_CORPUS_CONFIG)
    window = window_spec_from_config(load_data(DMF_DATA_CONFIG))
    _, tune_keys = pool_keys(sim_cfg, smoke)
    check_corpus(corpus_root, sim_cfg, tune_keys)
    manifest_path = record_dir / "fit_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["pools"]["tune"]["sha256"] != key_list_sha256(tune_keys):
        raise ValueError(f"{manifest_path} was fitted on a different tune pool")
    for name in models:
        if name not in manifest["models"]:
            raise ValueError(f"{name} is not in {manifest_path}; fit it first")
        model_dir = out_root / name
        record = calibrate_pad_vz(model_dir, corpus_root, window, tune_keys, log=log)
        entry = manifest["models"][name]
        entry["conformal_padvz"] = record
        entry["meta_sha256"] = sha256_file(model_dir / "meta.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"[calibrate] updated {manifest_path}")
    return manifest


def _parent(name: str) -> str:
    """Return the fitted model a directory name belongs to (``tcn_quantile_seed0`` -> ...)."""
    head, sep, tail = name.rpartition("_seed")
    return head if sep and tail.isdigit() and head in FORECASTERS else name


def seed_export_name(model: str, seed: int) -> str:
    """Return the directory name of a secondary seed export, e.g. ``tcn_quantile_seed0``."""
    return f"{model}_seed{seed}"


def export_seed(
    model: str,
    seed: int,
    *,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    record_dir: Path = DEFAULT_RECORD_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """Export a **non-selected** SGD seed as a self-contained, secondary model directory.

    The results-skeptic seed-sensitivity arm (M5). Same procedure as the deployed model --
    the seed's best-epoch checkpoint from ``<model>/seeds/``, the same train-partition
    normalisation (asserted byte-identical to the primary's), ONNX export and dmf parity,
    and, for an interval model, the tune-pool pad-v_z calibration -- written to
    ``<out>/<model>_seed<k>/`` and recorded in the manifest with
    ``role = "seed-sensitivity, not selected"``. The pre-registered selection and the
    primary directory are not touched.

    Args:
        model: An SGD model name, e.g. ``"tcn_quantile"``.
        seed: A fitted seed that is **not** the selected one.
        corpus_root: dmf corpus root.
        out_root: Artifact root.
        record_dir: Where ``fit_manifest.json`` lives.
        log: Progress sink.

    Returns:
        The manifest entry written.

    Raises:
        ValueError: If the model is not an SGD model in the manifest, the seed is the
            selected one or was never fitted, the pools differ, or the normalisation would
            differ from the primary's.
    """
    spec = FORECASTERS.get(model)
    if spec is None or spec.kind != "sgd":
        raise ValueError(f"{model!r} is not an SGD forecaster")
    manifest_path = record_dir / "fit_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = manifest["models"].get(model)
    if primary is None:
        raise ValueError(f"{model} is not in {manifest_path}; fit it first")
    if seed == primary["selected_seed"]:
        raise ValueError(f"seed {seed} is {model}'s selected seed; it is the primary directory")
    seeds_dir = out_root / model / "seeds"
    records = _seed_records(seeds_dir)
    if seed not in records:
        raise ValueError(f"{model} seed {seed} was never fitted ({sorted(records)} on disk)")
    record = records[seed]
    sim_cfg = load_sim(DMF_CORPUS_CONFIG)
    data_cfg = load_data(DMF_DATA_CONFIG)
    window = window_spec_from_config(data_cfg)
    train_keys, tune_keys = pool_keys(sim_cfg)
    for part, keys in (("train", train_keys), ("tune", tune_keys)):
        if manifest["pools"][part]["sha256"] != key_list_sha256(keys):
            raise ValueError(f"{manifest_path} was fitted on a different {part} pool")
    check_corpus(corpus_root, sim_cfg, train_keys + tune_keys)
    train_ds, tune_ds = _datasets(corpus_root, data_cfg, train_keys, tune_keys)
    cfg = load_model(spec.model_config)
    net = build_model(cfg, window, len(train_ds.input_columns), len(train_ds.target_columns))
    checkpoint = seeds_dir / str(record["checkpoint"])
    if sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint} does not match its seed record")
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    name = seed_export_name(model, seed)
    model_dir = out_root / name
    deployed = _write_deployed(
        spec, cfg, net, data_cfg, window, train_ds, tune_ds, model_dir, seed, None,
        name=name, role=SECONDARY_ROLE,
    )  # fmt: skip
    if deployed["norm_stats_sha256"] != primary["norm_stats_sha256"]:
        raise ValueError(f"{name}: normalisation differs from the primary {model}'s")
    if model in INTERVAL_MODELS:
        deployed["conformal_padvz"] = calibrate_pad_vz(
            model_dir, corpus_root, window, tune_keys, log=log
        )
        deployed["meta_sha256"] = sha256_file(model_dir / "meta.json")
    entry: dict[str, Any] = {
        "role": SECONDARY_ROLE,
        "parent": model,
        "seed": seed,
        "kind": spec.kind,
        "dmf_name": cfg.name,
        "head": cfg.head,
        "model_config": str(spec.model_config.relative_to(DMF_ROOT)),
        "fitted_on": FITTED_ON,
        "tuned_on": TUNED_ON,
        "selected_seed_of_parent": primary["selected_seed"],
        "tune_loss": float(record["tune_loss"]),
        "tune_loss_name": record["tune_loss_name"],
        "best_epoch": record["best_epoch"],
        "epochs_run": record["epochs_run"],
        "checkpoint": f"{model}/seeds/{record['checkpoint']}",
        "checkpoint_sha256": record["checkpoint_sha256"],
        **deployed,
    }
    manifest["models"][name] = entry
    manifest["models"] = {k: manifest["models"][k] for k in sorted(manifest["models"])}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(
        f"[export] {name}: tune {entry['tune_loss_name']} {entry['tune_loss']:.6f}; parity "
        f"max_abs_err {deployed['parity']['worst']['max_abs_err']:.2e}; wrote {model_dir}"
    )
    return entry


def _ort_version() -> str:
    """Return the installed onnxruntime version string."""
    import onnxruntime as ort

    return str(ort.__version__)


def corpus_frame(corpus_root: Path, key: RealizationKey, columns: Sequence[str]) -> pd.DataFrame:
    """Read one realization's corpus columns (full scale, 10 Hz, float32 on disk).

    Args:
        corpus_root: dmf corpus root.
        key: Realization key.
        columns: Corpus column names, e.g. :data:`rld.deck.bridge.FORECAST_CHANNELS`.

    Returns:
        The requested columns plus ``t`` (absolute full-scale seconds).
    """
    path = corpus_root / realization_path(spec_for_key(key))
    return pd.read_parquet(path, columns=["t", *columns])
