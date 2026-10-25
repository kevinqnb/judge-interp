"""Runtime, token and memory measurement shared by the representation and Clef tasks.

What is measured, per unit (one judge prompt for the representation tasks, one Clef record for
the Clef tasks), at batch size 1:

- ``seconds_*``: wall time of a phase, with ``torch.cuda.synchronize()`` before both clock reads
  on a CUDA device (without it the clock reads kernel launch, not compute).
- ``peak_alloc_bytes`` / ``peak_reserved_bytes``: ``torch.cuda.max_memory_allocated`` /
  ``max_memory_reserved`` over the unit, after ``reset_peak_memory_stats``. They include the
  model weights; ``weights_bytes`` (``memory_allocated`` after the model is loaded) is recorded
  once per run so the activation cost is peak minus weights.
- input tokens only. Neither judge generates text: the representation tasks prefill one prompt
  and read hidden states, Clef reads two logits per question. Output tokens are therefore 0 by
  construction, and ``n_questions`` / ``n_spans`` say how many verdicts or entities a unit yields.

On a CPU device the GPU-memory fields are ``None`` (never 0). Host peak RSS is
``resource.getrusage`` ``ru_maxrss`` of the process (high-water mark, so it includes model load).

Storage cost of a representation layer: ``rep_bytes`` (float32 in memory, which is what is stored;
the forward runs in the config's dtype) and the compressed size inside the ``.npz``.
"""
from __future__ import annotations

import os
import platform
import resource
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

PERCENTILE = 95  # reported alongside mean / median of every per-unit quantity


def is_cuda(device: str) -> bool:
    """Whether ``device`` is a CUDA device string; a CUDA device must actually be available."""
    cuda = device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    return cuda


class Meter:
    """Times phases and records GPU memory peaks on one device.

    Args:
        device: torch device string of the judge model.
    """

    def __init__(self, device: str):
        self.device = device
        self.cuda = is_cuda(device)

    def now(self) -> float:
        """Seconds on a monotonic clock, after all queued GPU work has finished."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def reset_peaks(self) -> None:
        """Start a new peak-memory window (no-op on CPU, where the peaks are ``None``)."""
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def peaks(self) -> dict:
        """``peak_alloc_bytes`` and ``peak_reserved_bytes`` since ``reset_peaks`` (``None`` on CPU)."""
        if not self.cuda:
            return {"peak_alloc_bytes": None, "peak_reserved_bytes": None}
        return {
            "peak_alloc_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def allocated_bytes(self) -> int | None:
        """Bytes of GPU memory currently allocated (``None`` on CPU)."""
        return int(torch.cuda.memory_allocated(self.device)) if self.cuda else None


def peak_rss_bytes() -> int:
    """High-water resident set size of this process in bytes (Linux ``ru_maxrss`` is in KiB)."""
    assert platform.system() == "Linux", "ru_maxrss units differ off Linux"
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def hardware(device: str) -> dict:
    """Strings and numbers that decide whether two runs' timings are comparable.

    Args:
        device: torch device string of the judge model.

    Returns:
        ``device``, torch / CUDA versions, the GPU name and total memory (``None`` on CPU), the host
        name and the CPU count visible to the process.
    """
    cuda = is_cuda(device)
    props = torch.cuda.get_device_properties(device) if cuda else None
    return {
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": props.name if cuda else None,
        "gpu_total_memory_bytes": int(props.total_memory) if cuda else None,
        "gpu_compute_capability": f"{props.major}.{props.minor}" if cuda else None,
        "host": platform.node(),
        "python": platform.python_version(),
        "cpu_count": len(os.sched_getaffinity(0)),
    }


def column_stats(prefix: str, values) -> dict:
    """Total, mean, median and p95 of a per-unit quantity as flat metrics.

    Args:
        prefix: metric-name prefix, e.g. ``"eff_trace_seconds"``.
        values: one number per unit, none ``None``.

    Returns:
        ``{prefix}_total``, ``_mean``, ``_median`` and ``_p95``.
    """
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and len(v) > 0 and np.isfinite(v).all(), (prefix, v.shape)
    return {
        f"{prefix}_total": float(v.sum()),
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_p95": float(np.percentile(v, PERCENTILE)),
    }


def peak_stats(prefix: str, values) -> dict:
    """Max, mean and median of a per-unit peak-memory column (``None`` for every stat on CPU).

    Args:
        prefix: metric-name prefix, e.g. ``"eff_gpu_peak_alloc_bytes"``.
        values: one int per unit, or ``None`` for each unit (CPU); never a mix.
    """
    if all(x is None for x in values):
        return {f"{prefix}_max": None, f"{prefix}_mean": None, f"{prefix}_median": None}
    assert all(x is not None for x in values), prefix
    v = np.asarray(values, dtype=np.float64)
    return {f"{prefix}_max": float(v.max()), f"{prefix}_mean": float(v.mean()), f"{prefix}_median": float(np.median(v))}


def unit_columns(units: list[dict]) -> dict[str, np.ndarray]:
    """Stack per-unit measurement dicts into arrays; GPU peaks are ``-1`` on CPU (see the ``.json``).

    Args:
        units: dicts with the same keys; values are numbers or ``None``.

    Returns:
        One array per key, in unit order.
    """
    assert units and all(set(u) == set(units[0]) for u in units), "units have different measurement keys"
    out = {}
    for k in units[0]:
        col = [u[k] for u in units]
        out[k] = np.asarray([-1 if x is None else x for x in col])
    return out


def npz_member_sizes(path: Path) -> dict[str, tuple[int, int]]:
    """Uncompressed and compressed byte size of every array in an ``.npz``.

    Args:
        path: a ``numpy.savez`` / ``savez_compressed`` file.

    Returns:
        ``{array name: (file_size, compress_size)}``. ``file_size`` includes the ``.npy`` header.
    """
    with zipfile.ZipFile(path) as z:
        return {Path(i.filename).stem: (i.file_size, i.compress_size) for i in z.infolist()}
