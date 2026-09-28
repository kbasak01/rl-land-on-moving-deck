"""Process liveness and CPU accounting from the standard library (Linux ``/proc``).

``psutil`` is not a declared dependency of this project, so the facts the run tracker, the
scheduler and the reconciler need are read directly: whether a pid is alive (and not a
zombie), when a process started, whether a live pid can be the process that wrote a given
``status.json`` (after a host restart pids are reused), and the CPU time of a process tree
(the learner plus its ``SubprocVecEnv`` workers).

Units: CPU time in seconds of CPU (user + system), summed over processes; start times in
Unix seconds (host clock).
"""

import os
import socket
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["pid_alive", "proc_start_unix", "status_owner_alive", "tree_cpu_seconds"]

_PROC = Path("/proc")

#: Slack between a process's recorded start and a status write, host seconds. A writer
#: process must have started before it wrote; clock-tick rounding of ``/proc`` start times
#: is 10 ms, and ISO timestamps are truncated to whole seconds.
_START_SLACK_S: float = 2.0


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


def proc_start_unix(pid: int) -> float | None:
    """Return when a process started, Unix seconds, or ``None`` if unknown.

    Args:
        pid: A process id.

    Returns:
        ``btime`` (``/proc/stat``) + ``starttime`` ticks (``/proc/<pid>/stat`` field 22) /
        ``SC_CLK_TCK``; ``None`` if the process or ``/proc`` is not there.
    """
    fields = _stat_fields(int(pid))
    if fields is None or len(fields) < 20:
        return None
    try:
        btime = next(
            float(line.split()[1])
            for line in (_PROC / "stat").read_text().splitlines()
            if line.startswith("btime ")
        )
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    # fields[0] is field 3 (state), so field 22 (starttime) is fields[19].
    return btime + float(fields[19]) / float(os.sysconf("SC_CLK_TCK"))


def _iso_unix(text: object) -> float | None:
    """Parse an ISO-8601 timestamp to Unix seconds, or ``None``."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def status_owner_alive(status: Mapping[str, Any]) -> bool:
    """Return whether the process that wrote a ``status.json`` can still be running.

    A plain :func:`pid_alive` is not enough after a host restart: the dead run's pid may
    have been given to an unrelated process. The writer is taken to be alive only if its
    pid is alive **and** that process started no later than the status's ``updated`` time
    (a process cannot have written before it started). A status written on another host
    is reported alive (this host cannot judge it).

    Args:
        status: A parsed ``status.json`` (``pid``, ``updated``, ``host``).

    Returns:
        True if the writer may be alive.
    """
    host = status.get("host")
    if host not in (None, socket.gethostname()):
        return True
    pid = status.get("pid")
    if not pid_alive(pid):
        return False
    updated = _iso_unix(status.get("updated"))
    started = proc_start_unix(int(str(pid)))
    if updated is None or started is None:
        return True
    return started <= updated + _START_SLACK_S


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
