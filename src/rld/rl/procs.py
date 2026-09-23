"""Process liveness and CPU accounting from the standard library (Linux ``/proc``).

``psutil`` is not a declared dependency of this project, so the two facts the run tracker
and the scheduler need are read directly: whether a pid is alive (and not a zombie), and
the CPU time of a process tree (the learner plus its ``SubprocVecEnv`` workers).

Units: CPU time in seconds of CPU (user + system), summed over processes.
"""

import os
from pathlib import Path

__all__ = ["pid_alive", "tree_cpu_seconds"]

_PROC = Path("/proc")


def _stat_fields(pid: int) -> list[str] | None:
    """Return ``/proc/<pid>/stat`` fields after the command name, or ``None``."""
    try:
        text = (_PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    # The command name is parenthesised and may contain spaces; split after the last ')'.
    return text[text.rfind(")") + 2 :].split()


def pid_alive(pid: object) -> bool:
    """Return whether a process exists and is not a zombie.

    Args:
        pid: A process id (anything ``int()`` accepts); ``None`` is not alive.

    Returns:
        True if the process exists and its state is not ``Z``/``X``.
    """
    try:
        value = int(pid)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    fields = _stat_fields(value)
    if fields is None:
        return True  # no /proc (not Linux): the kill probe is all we have
    return fields[0] not in ("Z", "X")


def tree_cpu_seconds(root: int | None = None) -> float:
    """Return the user + system CPU time of a process and all its live descendants.

    Args:
        root: Root pid; the calling process when ``None``.

    Returns:
        CPU seconds. Descendants that already exited are not included (their time is only
        visible to the parent after it reaps them).
    """
    root = os.getpid() if root is None else int(root)
    tick = float(os.sysconf("SC_CLK_TCK"))
    parent_of: dict[int, int] = {}
    cpu: dict[int, float] = {}
    for entry in _PROC.iterdir():
        if not entry.name.isdigit():
            continue
        fields = _stat_fields(int(entry.name))
        if fields is None or len(fields) < 13:
            continue
        pid = int(entry.name)
        parent_of[pid] = int(fields[1])
        cpu[pid] = (float(fields[11]) + float(fields[12])) / tick
    tree = {root}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_of.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return float(sum(cpu.get(pid, 0.0) for pid in tree))
