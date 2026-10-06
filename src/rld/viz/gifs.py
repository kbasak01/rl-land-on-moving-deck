"""Landing GIFs: several methods flown on the same frozen episode, side by side (Phase 9).

Each panel is a fresh re-flight of one committed episode row:

1. the listed episode is read from ``results/episodes/<regime>.parquet``;
2. the environment is built exactly as :func:`rld.eval.envs.make_env` builds it, plus
   ``visual_shapes=True`` so the drone draws as its mesh (P5-D5: visual shapes are not part
   of the dynamics world);
3. the per-step loop is the one in :func:`rld.eval.runner.run_chunk`;
4. after the episode, every :data:`rld.eval.runner.RECORD_COLUMNS` value is formatted the way
   the committed CSVs are and compared, as text, with the committed row
   (``results/e01/episodes.csv`` for classical controllers, ``results/e07/matrix/
   episodes.csv.gz`` for learned runs). Any difference raises :class:`ReflightMismatchError`.
   A GIF is never relabelled to fit a re-flight.

Rendering only reads state: ``getCameraImage`` (PyBullet's CPU TinyRenderer, DIRECT client)
never steps the simulation. Two visual-only bodies are touched between control steps:

- the ground plane is recoloured to read as sea;
- the painted pad disc is moved onto the plate. ``DeckPlatform._spawn_pad_marker`` places it
  once at spawn and ``advance`` never moves it, so without this it would hang where the deck
  was at reset (P9-D1). It carries no collision shape, so moving it cannot change a contact.

The GIFs are illustrations, hand-picked for visual clarity (P9-D1). They are not evidence and
not a sample of the distribution; the tables are the evidence.

Units: time in seconds model scale (×5 for full scale at λ = 1/25); lengths in metres model
scale; closing speed in m/s along the deck normal; tilt in degrees.
"""

import gzip
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pybullet as pyb

from rld.eval.envs import EvalConfigs, motion_for, pad_offset_for
from rld.eval.episodes import ListedEpisode, read_list
from rld.eval.runner import RECORD_COLUMNS, PolicySpec, _check_start, controller_spec

__all__ = [
    "Camera",
    "Case",
    "Flight",
    "Gallery",
    "PanelSpec",
    "ReflightMismatchError",
    "committed_row",
    "compose_gif",
    "fly",
    "load_gallery",
    "policy_spec",
    "record_text",
    "render_case",
    "seed_agreement",
]

#: Methods whose committed rows live in ``results/e01/episodes.csv``.
CLASSICAL_METHODS: frozenset[str] = frozenset(
    {"pid_track_descend", "pid_feedforward", "pid_feedforward_lowvz", "gated", "oracle_gated"}
)

SEA_RGBA = (0.16, 0.30, 0.42, 1.0)
SKY_RGB = (226, 234, 240)
SUCCESS_RGB = (22, 120, 60)
FAILURE_RGB = (176, 40, 38)
INK = (20, 20, 20)
INK_2 = (82, 81, 78)
SURFACE = (252, 252, 251)


class ReflightMismatchError(AssertionError):
    """A re-flight did not reproduce its committed episode row."""


@dataclass(frozen=True)
class PanelSpec:
    """One panel: a method and its training seed (0 for a deterministic controller).

    Attributes:
        method: Method label as in the committed tables.
        seed: Training seed of a learned run; ``0`` (the committed ``run_seed``) for a
            classical controller.
    """

    method: str
    seed: int = 0


@dataclass(frozen=True)
class Case:
    """One GIF: a listed episode and the panels flown on it.

    Attributes:
        slug: Output file stem.
        regime: Episode-list regime, e.g. ``"id"``.
        ss: Sea state, e.g. ``"SS5"``.
        index: Row index within the (regime, ss) cell of the frozen list.
        panels: Panels, left to right.
        caption: One line printed under the header.
    """

    slug: str
    regime: str
    ss: str
    index: int
    panels: tuple[PanelSpec, ...]
    caption: str = ""


@dataclass(frozen=True)
class Camera:
    """A fixed world-frame camera (PyBullet yaw/pitch convention, degrees).

    Attributes:
        target_m: Look-at point, metres model scale, world frame.
        distance_m: Distance from the target, metres.
        yaw_deg: Yaw about world z, degrees.
        pitch_deg: Pitch, degrees (negative looks down).
        fov_deg: Vertical field of view, degrees.
        width_px: Image width, pixels.
        height_px: Image height, pixels.
        supersample: Render at this multiple of the size, then downsample (anti-aliasing).
    """

    target_m: tuple[float, float, float] = (0.0, 0.0, 1.40)
    distance_m: float = 1.45
    yaw_deg: float = 38.0
    pitch_deg: float = -7.0
    fov_deg: float = 50.0
    width_px: int = 400
    height_px: int = 420
    supersample: int = 2

    def grab(self, client: int) -> np.ndarray:
        """Render one RGB frame from ``client``.

        Args:
            client: PyBullet physics client id.

        Returns:
            ``(height_px, width_px, 3)`` uint8.
        """
        view = pyb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(self.target_m),
            distance=self.distance_m,
            yaw=self.yaw_deg,
            pitch=self.pitch_deg,
            roll=0.0,
            upAxisIndex=2,
            physicsClientId=client,
        )
        proj = pyb.computeProjectionMatrixFOV(
            fov=self.fov_deg,
            aspect=self.width_px / self.height_px,
            nearVal=0.05,
            farVal=30.0,
            physicsClientId=client,
        )
        w, h = self.width_px * self.supersample, self.height_px * self.supersample
        _, _, rgba, _, _ = pyb.getCameraImage(
            w,
            h,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pyb.ER_TINY_RENDERER,
            shadow=1,
            lightDirection=[1.0, -0.6, 2.0],
            physicsClientId=client,
        )
        img = np.array(np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3])
        img[np.all(img >= 254, axis=2)] = SKY_RGB  # TinyRenderer's empty background
        if self.supersample == 1:
            return np.ascontiguousarray(img)
        from PIL import Image

        small = Image.fromarray(np.ascontiguousarray(img)).resize(
            (self.width_px, self.height_px), Image.Resampling.LANCZOS
        )
        return np.asarray(small, dtype=np.uint8)


@dataclass
class Flight:
    """One re-flown panel.

    Attributes:
        spec: The panel.
        frames: Rendered frames, one per ``frame_stride`` control steps plus the last.
        times_s: Episode time of each frame, seconds model scale.
        heights_m: Drone height above the pad point of each frame, metres model scale.
        record: ``EpisodeRecord.as_row()`` after the episode.
        committed: The committed row, as text.
    """

    spec: PanelSpec
    frames: list[np.ndarray] = field(default_factory=list)
    times_s: list[float] = field(default_factory=list)
    heights_m: list[float] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)
    committed: dict[str, str] = field(default_factory=dict)


def _fmt(value: Any) -> str:
    """Format one value the way the committed episode CSVs do (``repr`` floats)."""
    if value is None:
        return "nan"
    if isinstance(value, bool | np.bool_):
        return "True" if bool(value) else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return str(value)


def record_text(record: dict[str, Any]) -> dict[str, str]:
    """Return a flown episode's :data:`RECORD_COLUMNS` as committed-CSV text.

    Args:
        record: ``EpisodeRecord.as_row()``.

    Returns:
        Column to text. ``n_contacts`` is ``"0"`` without a touchdown, as the runner writes it.
    """
    out = {c: _fmt(record[c]) for c in RECORD_COLUMNS}
    if record["td_t_episode_s"] is None:
        out["n_contacts"] = "0"
    return out


def committed_row(
    results_dir: Path, regime: str, ss: str, index: int, method: str, seed: int
) -> dict[str, str]:
    """Return one committed episode row, every value as its committed text.

    Args:
        results_dir: The repository's ``results/`` directory.
        regime: Regime.
        ss: Sea state.
        index: Row index in the frozen list cell.
        method: Method label.
        seed: ``run_seed``.

    Returns:
        Column to text.

    Raises:
        KeyError: If the row is not committed (or is committed more than once).
    """
    if method in CLASSICAL_METHODS:
        text = (results_dir / "e01" / "episodes.csv").read_text(encoding="utf-8")
    else:
        with gzip.open(results_dir / "e07" / "matrix" / "episodes.csv.gz", "rt") as fh:
            text = fh.read()
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    hit = frame[
        (frame["regime"] == regime)
        & (frame["ss"] == ss)
        & (frame["index"] == str(index))
        & (frame["method"] == method)
        & (frame["run_seed"] == str(seed))
        & (frame["pad"] == "aft")
    ]
    if len(hit) != 1:
        raise KeyError(f"{regime}/{ss}/{index} {method} seed {seed}: {len(hit)} committed rows")
    return {str(k): str(v) for k, v in hit.iloc[0].items()}


def policy_spec(panel: PanelSpec, runs_dir: Path) -> PolicySpec:
    """Return the evaluation spec of a panel, built exactly as the committed flights built it.

    Args:
        panel: The panel.
        runs_dir: ``artifacts/runs`` (learned checkpoints: ``<method>/<seed>/final``).

    Returns:
        The registry controller's spec, or :func:`rld.eval.learned.learned_spec` of the run.
    """
    if panel.method in CLASSICAL_METHODS:
        return controller_spec(panel.method)
    from rld.eval.learned import inspect_run, learned_spec

    return learned_spec(inspect_run(panel.method, runs_dir / panel.method / str(panel.seed)))


def _listed(regime: str, ss: str, index: int, episodes_dir: Path) -> ListedEpisode:
    """Return the listed episode (regime, ss, index)."""
    rows = [
        r for r in read_list(episodes_dir / f"{regime}.parquet") if r.ss == ss and r.index == index
    ]
    if len(rows) != 1:
        raise KeyError(f"{regime}/{ss}/{index}: {len(rows)} listed rows")
    return rows[0]


def fly(
    case: Case,
    panel: PanelSpec,
    cfgs: EvalConfigs,
    *,
    results_dir: Path,
    runs_dir: Path,
    camera: Camera,
    frame_stride: int = 2,
) -> Flight:
    """Re-fly one committed episode, rendering a frame every ``frame_stride`` control steps.

    Args:
        case: The GIF case (which listed episode).
        panel: Method and seed.
        cfgs: The committed configs.
        results_dir: ``results/`` (episode lists and committed rows).
        runs_dir: ``artifacts/runs``.
        camera: The camera.
        frame_stride: Control steps per frame (30 Hz control: 2 gives 15 fps in real
            model-scale time).

    Returns:
        The :class:`Flight`, checked against its committed row.

    Raises:
        ReflightMismatchError: If any :data:`RECORD_COLUMNS` value differs from the committed
            row's text.
        ValueError: If the method consumes the ship-motion feed (not supported here).
    """
    from rld.envs.landing_env import DeckLandingAviary

    listed = _listed(case.regime, case.ss, case.index, results_dir / "episodes")
    spec = policy_spec(panel, runs_dir)
    if spec.needs_motion_feed or spec.privileged:
        raise ValueError(f"{panel.method}: feed-consuming or privileged methods are not rendered")
    motion = motion_for(
        cfgs,
        listed.vessel,
        listed.ss,
        listed.heading_deg,
        listed.speed_kn,
        listed.realization_seed,
        kind="jonswap",
        episode_seed=listed.episode_seed,
    )
    offset = pad_offset_for(cfgs, listed.vessel, listed.pad)
    env = DeckLandingAviary(
        motion=motion,
        pad=listed.pad,
        pad_radius_m=cfgs.pads.radius_model_m,
        pad_offset_m=offset,
        cfg=cfgs.landing,
        success=cfgs.success,
        obs_cfg=cfgs.observation,
        noise_cfg=cfgs.noise,
        reward_cfg=cfgs.reward,
        episode_seed=listed.episode_seed,
        visual_shapes=True,
    )
    flight = Flight(spec=panel)
    try:
        policy = spec.build(cfgs)
        env.set_motion(motion, listed.pad, offset)
        obs, info = env.reset(seed=listed.episode_seed)
        _check_start(env, info, listed, listed.pad, offset)
        client = int(env.CLIENT)
        plate = int(env._platform.body_id)
        marker = plate + 1  # spawned immediately after the plate (DeckPlatform.spawn)
        if pyb.getBodyInfo(marker, physicsClientId=client) is None:
            raise RuntimeError("pad marker body not found")
        pyb.changeVisualShape(
            int(env.PLANE_ID), -1, rgbaColor=list(SEA_RGBA), physicsClientId=client
        )
        policy.reset(listed.episode_seed, None)
        dt = 1.0 / float(cfgs.landing.ctrl_freq_hz)

        def snap(k: int) -> None:
            pos, quat = pyb.getBasePositionAndOrientation(plate, physicsClientId=client)
            normal = np.asarray(pyb.getMatrixFromQuaternion(quat)).reshape(3, 3)[:, 2]
            lift = np.asarray(pos) + 0.0012 * normal
            pyb.resetBasePositionAndOrientation(marker, lift.tolist(), quat, physicsClientId=client)
            flight.frames.append(camera.grab(client))
            flight.times_s.append(k * dt)
            flight.heights_m.append(float(np.asarray(env.pos[0])[2] - pos[2]))

        max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
        snap(0)
        k = 0
        for k in range(1, max_steps + 1):
            action = np.asarray(policy.act(obs), dtype=np.float64)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            if done or k % frame_stride == 0:
                snap(k)
            if done:
                break
        flight.record = dict(env.record.as_row())
    finally:
        env.close()
    flight.committed = committed_row(
        results_dir, case.regime, case.ss, case.index, panel.method, panel.seed
    )
    flown = record_text(flight.record)
    diffs = {
        c: (flown[c], flight.committed[c])
        for c in RECORD_COLUMNS
        if flown[c] != flight.committed[c]
    }
    if diffs:
        raise ReflightMismatchError(
            f"{case.slug} {panel.method} seed {panel.seed}: re-flight differs from the "
            f"committed row in {diffs}"
        )
    return flight


def _font(size: int) -> Any:
    """Matplotlib's bundled DejaVu Sans at ``size`` px (it has the λ and ° glyphs)."""
    import matplotlib
    from PIL import ImageFont

    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size=size)


def _label(method: str, seed: int) -> str:
    """Panel title: method, plus its training seed for a learned run."""
    return method if method in CLASSICAL_METHODS else f"{method} · seed {seed}"


def _outcome_text(committed: dict[str, str]) -> tuple[str, str]:
    """Return the final banner's two lines from the committed row."""
    outcome = committed["outcome"]
    vz = committed["rel_vz_normal_m_s"]
    tilt = committed["rel_tilt_deg"]
    head = outcome.replace("_", " ").upper()
    if vz in ("nan", ""):
        return head, "no touchdown"
    return head, f"closing speed {abs(float(vz)):.2f} m/s · rel. tilt {float(tilt):.1f}°"


def compose_gif(
    case: Case,
    flights: Sequence[Flight],
    out_gif: Path,
    *,
    header: str,
    fps: float,
    hold_s: float = 1.6,
    poster_png: Path | None = None,
) -> Path:
    """Compose synced panels into one looping GIF (and optionally a PNG poster frame).

    Panels share a clock: frame j of every panel is the same episode time. A panel whose
    episode has ended holds its last frame with the committed outcome banner.

    Args:
        case: The case.
        flights: One flight per panel, left to right.
        out_gif: Output GIF.
        header: Header line (episode identity).
        fps: Frames per second, so that playback runs at model-scale real time.
        hold_s: Extra display time of the last frame, seconds.
        poster_png: Optional output for the last composite frame.

    Returns:
        ``out_gif``.
    """
    from PIL import Image, ImageDraw

    pw, ph = flights[0].frames[0].shape[1], flights[0].frames[0].shape[0]
    top, bottom = 50, 22
    width, height = pw * len(flights), ph + top + bottom
    n = max(len(f.frames) for f in flights)
    f_head, f_small, f_panel, f_banner = _font(15), _font(11), _font(13), _font(15)
    composites: list[Image.Image] = []
    for j in range(n):
        canvas = Image.new("RGB", (width, height), SURFACE)
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), header, fill=INK, font=f_head)
        if case.caption:
            draw.text((10, 28), case.caption, fill=INK_2, font=f_small)
        for p, flight in enumerate(flights):
            i = min(j, len(flight.frames) - 1)
            x0 = p * pw
            canvas.paste(Image.fromarray(flight.frames[i]), (x0, top))
            draw.rectangle((x0 + 6, top + 6, x0 + pw - 6, top + 26), fill=(255, 255, 255))
            draw.text(
                (x0 + 11, top + 9),
                _label(flight.spec.method, flight.spec.seed),
                fill=INK,
                font=f_panel,
            )
            draw.text(
                (x0 + 11, top + ph - 20),
                f"t = {flight.times_s[i]:.2f} s   height above pad {flight.heights_m[i]:+.2f} m",
                fill=(255, 255, 255),
                font=f_small,
            )
            if j >= len(flight.frames) - 1:
                head, sub = _outcome_text(flight.committed)
                colour = SUCCESS_RGB if flight.committed["outcome"] == "success" else FAILURE_RGB
                draw.rectangle((x0 + 6, top + 32, x0 + pw - 6, top + 76), fill=colour)
                draw.text((x0 + 12, top + 36), head, fill=(255, 255, 255), font=f_banner)
                draw.text((x0 + 12, top + 57), sub, fill=(255, 255, 255), font=f_small)
            if p:
                draw.line((x0, top, x0, top + ph), fill=SURFACE, width=2)
        draw.text(
            (10, height - bottom + 5),
            "Simulation only · deck motion Froude-scaled at λ = 1/25 (1 s model = 5 s full "
            "scale) · real-time model-scale playback · illustrative, hand-picked episode",
            fill=INK_2,
            font=f_small,
        )
        composites.append(canvas)
    # One shared palette (no per-frame flicker), taken from first, middle and last frames.
    probe = Image.new("RGB", (width, height * 3))
    for k, idx in enumerate((0, n // 2, n - 1)):
        probe.paste(composites[idx], (0, k * height))
    palette = probe.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [c.quantize(palette=palette, dither=Image.Dither.NONE) for c in composites]
    durations = [round(1000.0 / fps)] * n
    durations[-1] += round(1000.0 * hold_s)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    if poster_png is not None:
        composites[-1].save(poster_png, optimize=True)
    return out_gif


@dataclass(frozen=True)
class Gallery:
    """The GIF gallery config (``configs/viz/gifs.yaml``).

    Attributes:
        cases: The cases, in README order.
        frame_stride: Control steps per frame.
        hold_s: Extra display time of the last frame, seconds.
    """

    cases: tuple[Case, ...]
    frame_stride: int
    hold_s: float


def load_gallery(path: Path) -> Gallery:
    """Read the gallery config.

    Args:
        path: ``configs/viz/gifs.yaml``.

    Returns:
        The :class:`Gallery`.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = tuple(
        Case(
            slug=str(c["slug"]),
            regime=str(c["regime"]),
            ss=str(c["ss"]),
            index=int(c["index"]),
            panels=tuple(PanelSpec(str(p["method"]), int(p.get("seed", 0))) for p in c["panels"]),
        )
        for c in raw["cases"]
    )
    return Gallery(cases=cases, frame_stride=int(raw["frame_stride"]), hold_s=float(raw["hold_s"]))


def seed_agreement(
    results_dir: Path, case: Case, panel: PanelSpec, outcome: str
) -> tuple[int, int]:
    """Count the method's committed seeds with ``outcome`` on the case's episode.

    Args:
        results_dir: ``results/``.
        case: The case.
        panel: The panel (its method).
        outcome: The outcome class shown.

    Returns:
        ``(k, n)``: k of the method's n committed runs on this episode end in ``outcome``
        (n = 1 for a deterministic controller).
    """
    if panel.method in CLASSICAL_METHODS:
        return (1, 1)
    frame = pd.read_csv(
        results_dir / "e07" / "matrix" / "episodes.csv.gz",
        usecols=["regime", "ss", "index", "pad", "method", "run_seed", "outcome"],
    )
    hit = frame[
        (frame["regime"] == case.regime)
        & (frame["ss"] == case.ss)
        & (frame["index"] == case.index)
        & (frame["method"] == panel.method)
        & (frame["pad"] == "aft")
    ]
    return int((hit["outcome"] == outcome).sum()), int(len(hit))


def render_case(
    case: Case,
    gallery: Gallery,
    cfgs: EvalConfigs,
    *,
    results_dir: Path,
    runs_dir: Path,
    out_dir: Path,
    camera: Camera | None = None,
) -> list[dict[str, Any]]:
    """Re-fly, check and render one case to ``<out_dir>/<slug>.gif`` and ``.png``.

    Args:
        case: The case.
        gallery: The gallery (frame stride, hold).
        cfgs: The committed configs.
        results_dir: ``results/``.
        runs_dir: ``artifacts/runs``.
        out_dir: Output directory.
        camera: Camera (default :class:`Camera`).

    Returns:
        One manifest row per panel: identity, the committed outcome, closing speed and
        relative tilt, and the k-of-n seed agreement.
    """
    cam = camera or Camera()
    flights = [
        fly(
            case,
            panel,
            cfgs,
            results_dir=results_dir,
            runs_dir=runs_dir,
            camera=cam,
            frame_stride=gallery.frame_stride,
        )
        for panel in case.panels
    ]
    listed = _listed(case.regime, case.ss, case.index, results_dir / "episodes")
    header = (
        f"{case.regime} · {case.ss} · list #{case.index} · {listed.vessel}, heading "
        f"{listed.heading_deg:g}°, {listed.speed_kn:g} kn · aft pad"
    )
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for flight in flights:
        outcome = flight.committed["outcome"]
        k, n = seed_agreement(results_dir, case, flight.spec, outcome)
        rows.append(
            {
                "slug": case.slug,
                "regime": case.regime,
                "ss": case.ss,
                "index": case.index,
                "vessel": listed.vessel,
                "heading_deg": listed.heading_deg,
                "speed_kn": listed.speed_kn,
                "method": flight.spec.method,
                "run_seed": flight.spec.seed,
                "outcome": outcome,
                "rel_vz_normal_m_s": flight.committed["rel_vz_normal_m_s"],
                "rel_tilt_deg": flight.committed["rel_tilt_deg"],
                "td_t_episode_s": flight.committed["td_t_episode_s"],
                "steps": flight.committed["steps"],
                "seeds_with_outcome": k,
                "seeds_flown": n,
                "reflight_matches_committed": True,
            }
        )
        if n > 1:
            notes.append(f"{flight.spec.method}: {k} of {n} seeds {outcome.replace('_', ' ')}")
    shown = Case(
        case.slug,
        case.regime,
        case.ss,
        case.index,
        case.panels,
        "Seeds on this episode: " + "; ".join(notes) + ". Outcomes from the committed rows.",
    )
    fps = float(cfgs.landing.ctrl_freq_hz) / gallery.frame_stride
    compose_gif(
        shown,
        flights,
        out_dir / f"{case.slug}.gif",
        header=header,
        fps=fps,
        hold_s=gallery.hold_s,
        poster_png=out_dir / f"{case.slug}.png",
    )
    return rows
