"""Probability calibration metrics and reliability diagrams for research evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE for binary probabilities in [0, 1].

    Bins predicted probabilities, then averages |confidence - accuracy|
    weighted by bin mass. Not a production guarantee — just a diagnostic.
    """
    diagram = reliability_bins(y_true, y_prob, n_bins=n_bins)
    return float(diagram.ece)


@dataclass
class ReliabilityDiagram:
    """Equal-width reliability bins for binary predicted probabilities."""

    n_bins: int
    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray
    ece: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "bin_edges": self.bin_edges.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "bin_count": self.bin_count.tolist(),
            "ece": self.ece,
        }


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityDiagram:
    """Compute equal-width reliability bins and ECE.

    Empty bins get NaN confidence/accuracy and zero count. ECE weights
    only non-empty bins by their mass.
    """
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    conf = np.full(n_bins, np.nan, dtype=float)
    acc = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    ece = 0.0
    n = y_true.size
    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = int(mask.sum())
        if not np.any(mask):
            continue
        conf[b] = float(y_prob[mask].mean())
        acc[b] = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc[b] - conf[b])

    return ReliabilityDiagram(
        n_bins=n_bins,
        bin_edges=edges,
        bin_confidence=conf,
        bin_accuracy=acc,
        bin_count=counts,
        ece=float(ece),
    )


def format_reliability_ascii(diagram: ReliabilityDiagram, width: int = 40) -> str:
    """Render an ASCII reliability table (no matplotlib required)."""
    lines = [
        f"Reliability bins (n_bins={diagram.n_bins}, ECE={diagram.ece:.4f})",
        f"{'bin':>6} {'lo':>6} {'hi':>6} {'conf':>8} {'acc':>8} {'n':>6}  gap",
    ]
    for b in range(diagram.n_bins):
        lo = diagram.bin_edges[b]
        hi = diagram.bin_edges[b + 1]
        n = int(diagram.bin_count[b])
        if n == 0:
            lines.append(f"{b:6d} {lo:6.2f} {hi:6.2f} {'—':>8} {'—':>8} {n:6d}")
            continue
        c = float(diagram.bin_confidence[b])
        a = float(diagram.bin_accuracy[b])
        gap = abs(a - c)
        bar_len = max(0, min(width, int(round(gap * width * 4))))
        bar = "#" * bar_len
        lines.append(
            f"{b:6d} {lo:6.2f} {hi:6.2f} {c:8.3f} {a:8.3f} {n:6d}  {bar} ({gap:.3f})"
        )
    return "\n".join(lines)


def save_reliability_diagram(
    diagram: ReliabilityDiagram,
    path: str | Path,
    title: str = "Reliability diagram",
) -> Path:
    """Save a reliability diagram PNG (matplotlib) or ASCII fallback.

    If matplotlib is unavailable, writes ``<path>.txt`` with ASCII bins
    and returns that path instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        txt_path = path.with_suffix(".txt")
        txt_path.write_text(format_reliability_ascii(diagram) + "\n", encoding="utf-8")
        return txt_path

    centers = 0.5 * (diagram.bin_edges[:-1] + diagram.bin_edges[1:])
    mask = diagram.bin_count > 0
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    if np.any(mask):
        ax.plot(
            diagram.bin_confidence[mask],
            diagram.bin_accuracy[mask],
            "o-",
            color="C0",
            label="model",
        )
        ax.bar(
            centers[mask],
            diagram.bin_count[mask] / max(diagram.bin_count.sum(), 1),
            width=0.08,
            alpha=0.25,
            color="C1",
            label="bin mass (norm)",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title(f"{title}\nECE={diagram.ece:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
