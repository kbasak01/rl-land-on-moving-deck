"""The sea-state curriculum: SS3 -> SS4 -> SS5, promoted on tune-pool success (P3-D1 section 6).

Rule (pre-registered): a stage is promoted when the rolling tune-pool success **at the
current sea state** is ``>= promote_success`` (0.80) over ``>= window_episodes`` (100)
episodes. SS6 is never a stage.

Where the outcomes come from. Each periodic evaluation flies the fixed P3-D3 tune-pool draw
(60 episodes per sea state with the committed ``configs/control/tuning.yaml``); only the
current stage's episodes enter the window. With 60 episodes per evaluation the window of
100 therefore always spans **two consecutive evaluations** -- two different checkpoints --
so a single lucky evaluation cannot promote. The window is cleared at every promotion,
because outcomes at the previous sea state say nothing about the new one.

The decision itself is the pure function :func:`should_promote`; :class:`Curriculum` only
holds the window and the stage index. Broadcasting the stage to the training workers is the
caller's job (``VecEnv.env_method("set_stage", ss)``; see :mod:`rld.rl.callbacks`).

Units: episode counts are dimensionless; ``steps`` are training env steps (1/30 s model
scale each).
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rld.rl.config import FORBIDDEN_SEA_STATES, TRAINING_SEA_STATES, CurriculumConfig

__all__ = ["Curriculum", "should_promote"]


def should_promote(window: Sequence[bool], *, threshold: float, min_episodes: int) -> bool:
    """Return whether a rolling window of successes meets the promotion rule.

    Args:
        window: Success flags of the most recent tune-pool episodes at the current stage,
            oldest first.
        threshold: Required success fraction, dimensionless (0.80).
        min_episodes: Required window length, episodes (100).

    Returns:
        True iff the window holds at least ``min_episodes`` outcomes and the success
        fraction over the most recent ``min_episodes`` is ``>= threshold``.
    """
    if min_episodes < 1 or len(window) < min_episodes:
        return False
    recent = list(window)[-min_episodes:]
    return sum(bool(x) for x in recent) >= threshold * min_episodes - 1e-9


@dataclass
class Curriculum:
    """The curriculum's state: the stage index and the rolling window at that stage.

    Attributes:
        cfg: The curriculum config.
        index: Current stage index into ``cfg.stages``.
        history: One record per promotion: ``{"steps", "from", "to", "window_success"}``.
    """

    cfg: CurriculumConfig
    index: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    _window: deque[bool] = field(init=False)

    def __post_init__(self) -> None:
        """Validate the stages and open an empty window.

        Raises:
            ValueError: If a stage is SS6 or not a training sea state, or the config is
                otherwise unusable.
        """
        stages = self.cfg.stages
        if not stages:
            raise ValueError("a curriculum needs at least one stage")
        if bad := sorted(set(stages) & set(FORBIDDEN_SEA_STATES)):
            raise ValueError(f"curriculum stage {bad} is never trained on")
        if bad := sorted(set(stages) - set(TRAINING_SEA_STATES)):
            raise ValueError(f"curriculum stages {bad} are not training sea states")
        if not 0 <= self.index < len(stages):
            raise ValueError(f"stage index {self.index} out of range")
        self._window = deque(maxlen=self.cfg.window_episodes)

    @property
    def stage(self) -> str:
        """Return the current stage's sea state."""
        return self.cfg.stages[self.index]

    @property
    def sampling_stage(self) -> str | None:
        """Return the sea state training resets sample from; ``None`` = all stages.

        ``None`` only when the curriculum is disabled.
        """
        return self.stage if self.cfg.enabled else None

    @property
    def is_final(self) -> bool:
        """Return whether the current stage is the last one."""
        return self.index == len(self.cfg.stages) - 1

    @property
    def window(self) -> tuple[bool, ...]:
        """Return the rolling window, oldest first."""
        return tuple(self._window)

    @property
    def window_success(self) -> float:
        """Return the window's success fraction, or NaN when empty."""
        return float("nan") if not self._window else sum(self._window) / len(self._window)

    def observe(self, successes: Sequence[bool], steps: int) -> bool:
        """Add one evaluation's current-stage outcomes and promote if the rule is met.

        Args:
            successes: Success flags of this evaluation's episodes **at the current
                stage's sea state**, in episode order.
            steps: Training env steps at this evaluation.

        Returns:
            True if the stage advanced.
        """
        if not self.cfg.enabled or self.is_final:
            self._window.extend(bool(s) for s in successes)
            return False
        self._window.extend(bool(s) for s in successes)
        if not should_promote(
            self._window,
            threshold=self.cfg.promote_success,
            min_episodes=self.cfg.window_episodes,
        ):
            return False
        old = self.stage
        success = self.window_success
        self.index += 1
        self._window.clear()
        self.history.append(
            {"steps": int(steps), "from": old, "to": self.stage, "window_success": success}
        )
        return True

    def snapshot(self) -> dict[str, Any]:
        """Return everything needed to continue: stage index, promotion history, the window.

        Unlike :meth:`state` (a summary for ``status.json``) this holds the window's
        individual success flags, so a resumed run promotes exactly when the uninterrupted
        one would have, given the same evaluation outcomes.
        """
        return {
            "stages": list(self.cfg.stages),
            "index": self.index,
            "history": [dict(h) for h in self.history],
            "window": [bool(x) for x in self._window],
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore a :meth:`snapshot`.

        Args:
            snap: From :meth:`snapshot` of a curriculum with the same stages.

        Raises:
            ValueError: If the stages differ or the index is out of range.
        """
        if tuple(snap["stages"]) != tuple(self.cfg.stages):
            raise ValueError(f"curriculum stages {snap['stages']} != {list(self.cfg.stages)}")
        index = int(snap["index"])
        if not 0 <= index < len(self.cfg.stages):
            raise ValueError(f"stage index {index} out of range")
        self.index = index
        self.history = [dict(h) for h in snap["history"]]
        self._window = deque((bool(x) for x in snap["window"]), maxlen=self.cfg.window_episodes)

    def state(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for ``status.json``."""
        return {
            "stage": self.stage,
            "index": self.index,
            "enabled": self.cfg.enabled,
            "window_n": len(self._window),
            "window_success": None if not self._window else self.window_success,
            "history": list(self.history),
        }
