#!/usr/bin/env python3
"""Produce a showcase visualization comparing Procrustes baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patheffects, colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Ensure we can import the project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from algorithms.baseline_procrustes import baseline_procrustes
from algorithms.piecewise_procrustes import piecewise_procrustes
from main_evaluation import kmeans_procrustes_baseline, piecewise_align

BG_COLOR = "#fdf8f3"
REFERENCE_COLOR = "#cfd5dd"
POINT_EDGE = "#fefefe"
SEGMENT_COLORS = ("#4c6ef5", "#ff6b6b", "#f6c343", "#2ec4b6", "#b47aea")
TRAJECTORY_CMAP = LinearSegmentedColormap.from_list(
    "procrustes_flow",
    ["#3f37c9", "#5850ec", "#b5179e", "#ff6b6b", "#ffb703"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a README-friendly comparison of Procrustes alignments.",
    )
    default_output = THIS_DIR / "assets" / "temporal_shift.png"
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Destination path for the PNG (default: {default_output})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed controlling the curved input trajectory.",
    )
    parser.add_argument(
        "--pieces",
        type=int,
        default=3,
        help="Number of segments (k) for both k-means and piecewise DP.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=4,
        help="Ambient dimension of the synthetic trajectory (must be >= 2).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in dots-per-inch.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also render an animated GIF showing the piecewise alignment unfolding.",
    )
    default_gif = default_output.with_suffix(".gif")
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=default_gif,
        help=f"Path for the optional GIF (default: {default_gif})",
    )
    return parser.parse_args()


def _styled_curve(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generates a smooth flower-like trajectory embedded in R^dim."""
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    radius = 1.0 + 0.18 * np.sin(3.4 * theta) + 0.08 * np.sin(8.5 * theta)
    x = radius * np.cos(theta + 0.25 * np.sin(2.0 * theta))
    y = 0.82 * radius * np.sin(theta) + 0.17 * np.sin(4.3 * theta)

    coords = np.zeros((n_samples, dim))
    coords[:, 0] = x
    coords[:, 1] = y
    for d in range(2, dim):
        phase = 0.8 * d
        coords[:, d] = 0.4 * np.sin((d + 1.3) * theta + phase)
    coords += 0.01 * rng.normal(size=coords.shape)
    return coords


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q *= signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _apply_rotation_perturbation(
    base: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if scale <= 0:
        return base
    dim = base.shape[0]
    noise = rng.normal(size=(dim, dim))
    skew = noise - noise.T
    delta = np.eye(dim) + scale * skew
    q_delta, r = np.linalg.qr(delta)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q_delta *= signs
    if np.linalg.det(q_delta) < 0:
        q_delta[:, 0] *= -1
    return base @ q_delta


def generate_showcase_data(
    *,
    n_samples: int = 360,
    dim: int = 4,
    n_pieces: int = 3,
    intracluster_drift: float = 0.006,
    intercluster_jump: float = 0.8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create a smooth but misaligned dataset ideal for visualization."""
    if n_pieces <= 0:
        raise ValueError("n_pieces must be positive.")
    rng = np.random.default_rng(seed)
    X = _styled_curve(n_samples, dim, rng)
    Y = np.empty_like(X)

    base_lengths = np.full(n_pieces, n_samples // n_pieces, dtype=int)
    base_lengths[: n_samples % n_pieces] += 1

    true_cutoffs: list[int] = []
    idx = 0
    Q_current = _random_rotation(dim, rng)
    for piece_idx, seg_len in enumerate(base_lengths):
        for _ in range(seg_len):
            Q_current = _apply_rotation_perturbation(Q_current, intracluster_drift, rng)
            Y[idx] = X[idx] @ Q_current
            idx += 1
        if piece_idx < n_pieces - 1:
            true_cutoffs.append(idx)
            Q_current = _apply_rotation_perturbation(Q_current, intercluster_jump, rng)
    return X, Y, true_cutoffs


def _segments_from_cutoffs(cutoffs: Sequence[int], length: int) -> np.ndarray:
    ordered = [c for c in sorted(cutoffs) if 0 <= c < length]
    bounds = [0] + ordered + [length]
    segments = np.zeros(length, dtype=int)
    for seg_id, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        segments[start:end] = seg_id
    return segments


def _ordered_labels(labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    ordering = np.argsort([np.mean(np.where(labels == val)[0]) for val in unique])
    mapped = np.zeros_like(labels)
    for new_idx, uniq_idx in enumerate(ordering):
        mapped[labels == unique[uniq_idx]] = new_idx
    return mapped


def _line_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 2))
    return np.stack([points[:-1], points[1:]], axis=1)


def _draw_gradient_path(
    ax,
    coords: np.ndarray,
    *,
    linewidth: float = 3.0,
    alpha: float = 0.96,
) -> None:
    norm = plt.Normalize(0, coords.shape[0] - 1)
    segments = _line_segments(coords)
    if len(segments):
        lc = LineCollection(
            segments,
            cmap=TRAJECTORY_CMAP,
            norm=norm,
            linewidths=linewidth,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
        )
        lc.set_array(np.arange(coords.shape[0] - 1))
        ax.add_collection(lc)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.linspace(0, 1, coords.shape[0]),
        cmap=TRAJECTORY_CMAP,
        s=16,
        linewidths=0.25,
        edgecolors=POINT_EDGE,
        zorder=3,
    )


def _draw_segmented_path(
    ax,
    coords: np.ndarray,
    segments: Sequence[int],
    *,
    linewidth: float = 3.0,
) -> None:
    """Render trajectory with a fixed color per segment."""
    if len(segments) != len(coords):
        raise ValueError("segments length must match coordinates.")
    for start, end, seg_id in _runs(segments):
        seg_coords = coords[start:end]
        if seg_coords.size == 0:
            continue
        color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]
        lines = _line_segments(seg_coords)
        if len(lines):
            ax.add_collection(
                LineCollection(
                    lines,
                    colors=[color],
                    linewidths=linewidth,
                    alpha=0.95,
                    capstyle="round",
                    joinstyle="round",
                )
            )
        ax.scatter(
            seg_coords[:, 0],
            seg_coords[:, 1],
            color=color,
            s=18,
            linewidths=0.3,
            edgecolors=POINT_EDGE,
            zorder=3,
        )


def _fit_pca_basis(data: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and basis vectors for the top principal components."""
    if data.size == 0:
        raise ValueError("Cannot fit PCA basis on empty data.")
    comps = min(n_components, data.shape[1])
    if comps <= 0:
        raise ValueError("n_components must be positive.")
    mean = data.mean(axis=0)
    centered = data - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:comps].T
    return mean, basis


def _project_to_basis(data: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project data into the provided PCA basis."""
    return (data - mean) @ basis


def _draw_error_spokes(ax, reference: np.ndarray, coords: np.ndarray) -> None:
    if reference is None or len(coords) == 0:
        return
    idxs = np.linspace(0, len(coords) - 1, 30, dtype=int)
    for idx in np.unique(idxs):
        ax.plot(
            [reference[idx, 0], coords[idx, 0]],
            [reference[idx, 1], coords[idx, 1]],
            color="#1f2933",
            linewidth=0.5,
            alpha=0.35,
            zorder=1,
        )


def _mark_boundaries(ax, coords: np.ndarray, cutoffs: Sequence[int]) -> None:
    length = coords.shape[0]
    for cutoff in cutoffs:
        if 0 < cutoff < length:
            pt = coords[cutoff]
            ax.scatter(
                pt[0],
                pt[1],
                s=150,
                facecolors="none",
                edgecolors="#0f172a",
                linewidths=1.5,
                zorder=4,
            )


def _render_panel(
    ax,
    coords: np.ndarray,
    *,
    title: str,
    reference: np.ndarray | None,
    cost: float | None = None,
    cutoffs: Sequence[int] | None = None,
    show_error: bool = False,
    segments: Sequence[int] | None = None,
    show_reference: bool = True,
) -> None:
    ax.set_facecolor("#ffffff")
    if reference is not None and show_reference:
        ax.plot(
            reference[:, 0],
            reference[:, 1],
            color=REFERENCE_COLOR,
            linewidth=2.4,
            zorder=0,
        )
    if segments is not None:
        _draw_segmented_path(ax, coords, segments)
    else:
        _draw_gradient_path(ax, coords)
    if show_error and reference is not None:
        _draw_error_spokes(ax, reference, coords)
    if cutoffs:
        _mark_boundaries(ax, coords, cutoffs)
    if cost is not None:
        ax.text(
            0.02,
            0.96,
            f"error: {cost:,.1f}",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="left",
            va="top",
            color="#111111",
            bbox={
                "facecolor": "#ffffff",
                "edgecolor": "#e2e2e2",
                "boxstyle": "round,pad=0.2",
                "alpha": 0.9,
            },
        )
    ax.set_title(title, fontsize=12.5, pad=8, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("#d5d5d5")


def _runs(sequence: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    if len(sequence) == 0:
        return
    start = 0
    current = sequence[0]
    for idx in range(1, len(sequence)):
        val = sequence[idx]
        if val != current:
            yield (start, idx, current)
            start = idx
            current = val
    yield (start, len(sequence), current)


def _plot_segments(ax, rows: Sequence[tuple[str, Sequence[int]]], length: int) -> None:
    ax.set_facecolor("#ffffff")
    for idx, (label, segments) in enumerate(rows):
        baseline = idx
        for start, end, seg_id in _runs(segments):
            width = max(end - start, 0)
            if width == 0:
                continue
            color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]
            ax.add_patch(
                Rectangle(
                    (start, baseline - 0.35),
                    width,
                    0.7,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.95,
                )
            )
        ax.text(
            -5,
            baseline,
            label,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#202124",
        )
    ax.set_xlim(0, length)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(np.linspace(0, length, 5, dtype=int))
    ax.set_xlabel("sample index", fontsize=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _segment_progress(points: np.ndarray, target: np.ndarray, segments: np.ndarray, progress: float) -> np.ndarray:
    """Linearly interpolate segment-wise so pieces snap into place sequentially."""
    n_segments = int(segments.max() + 1) if len(segments) else 1
    alpha = np.clip(progress - segments, 0.0, 1.0)
    return points + alpha[:, None] * (target - points)


def render_alignment_gif(
    raw_coords: np.ndarray,
    aligned_coords: np.ndarray,
    reference: np.ndarray,
    segments: np.ndarray,
    cutoffs: Sequence[int],
    *,
    output_path: Path,
    bounds: tuple[float, float, float, float],
    frames: int = 120,
    fps: int = 18,
    hold_seconds: float = 2.0,
    descriptor: str = "coords",
) -> None:
    """Animate the piecewise alignment unfolding segment by segment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_segments = int(segments.max() + 1)
    move_frames = max(1, frames)
    progress_values = np.linspace(0.0, n_segments, move_frames)
    hold_frames = max(1, int(hold_seconds * fps))
    progress_values = np.concatenate(
        [progress_values, np.full(hold_frames, n_segments, dtype=float)]
    )
    base_colors = [SEGMENT_COLORS[int(seg) % len(SEGMENT_COLORS)] for seg in segments]
    base_rgba = mcolors.to_rgba_array(base_colors)
    segments_arr = np.asarray(segments, dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 7.8), dpi=160)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"Piecewise alignment ({descriptor})",
        fontsize=14,
        weight="bold",
        y=0.98,
        color="#161616",
    )
    ax.set_facecolor("#ffffff")
    ax.plot(reference[:, 0], reference[:, 1], color=REFERENCE_COLOR, linewidth=2.2, zorder=0)
    _mark_boundaries(ax, reference, cutoffs)

    origin_rgba = base_rgba.copy()
    origin_rgba[:, 3] = 0.5
    origin_scatter = ax.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        color=origin_rgba,
        s=12,
        linewidths=0,
        zorder=1,
    )

    scatter = ax.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        color=base_rgba,
        s=18,
        linewidths=0.1,
        edgecolors=POINT_EDGE,
        zorder=3,
    )
    path_line, = ax.plot(
        raw_coords[:, 0],
        raw_coords[:, 1],
        color="#522a68",
        linewidth=1.2,
        alpha=0.4,
    )
    progress_text = fig.text(
        0.04,
        0.935,
        "",
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
        color="#111111",
    )
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)

    def _update(frame_idx: int):
        progress = progress_values[frame_idx]
        coords = _segment_progress(raw_coords, aligned_coords, segments, progress)
        seg_progress = np.clip(progress - segments_arr, 0.0, 1.0)
        rgba = base_rgba.copy()
        rgba[:, 3] = 0.2 + 0.8 * seg_progress
        scatter.set_offsets(coords)
        scatter.set_facecolors(rgba)
        sizes = 12 + 16 * seg_progress
        scatter.set_sizes(sizes)
        path_line.set_data(coords[:, 0], coords[:, 1])
        active = min(int(progress), n_segments - 1)
        progress_text.set_text(f"aligning piece {active + 1}/{n_segments}")
        origin_fade = base_rgba.copy()
        origin_fade[:, 3] = 0.5 * (1.0 - seg_progress)
        origin_scatter.set_facecolors(origin_fade)
        return scatter, path_line, origin_scatter, progress_text

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(progress_values),
        interval=1000 / fps,
        blit=True,
    )
    caption = (
        f"Top two PCA components"
    )
    fig.text(
        0.5,
        0.015,
        caption,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#494949",
    )
    writer = animation.PillowWriter(fps=fps, metadata={"loop": 0})
    anim.save(
        output_path,
        writer=writer,
        dpi=200,
        savefig_kwargs={"facecolor": BG_COLOR},
    )
    plt.close(fig)
    print(f"Saved animation to {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = 360
    if args.pieces <= 0:
        raise ValueError("--pieces must be positive.")
    n_pieces = int(args.pieces)
    dp_samples = 180
    if args.dim < 2:
        raise ValueError("--dim must be at least 2.")
    dim = int(args.dim)

    X, Y, true_cutoffs = generate_showcase_data(
        n_samples=n_samples,
        dim=dim,
        n_pieces=n_pieces,
        intracluster_drift=0.006,
        intercluster_jump=0.9,
        seed=args.seed,
    )

    order = np.arange(n_samples, dtype=int)
    Y_ordered = Y[order]
    pca_mean, pca_basis = _fit_pca_basis(Y_ordered, n_components=2)

    def _project(arr: np.ndarray) -> np.ndarray:
        return _project_to_basis(arr, pca_mean, pca_basis)

    plot_Y = _project(Y_ordered)
    plot_X = _project(X[order])
    true_segments = _segments_from_cutoffs(true_cutoffs, len(order))

    Q_global = baseline_procrustes(X, Y)
    global_aligned = (X @ Q_global)[order]
    global_cost = float(np.linalg.norm(global_aligned - Y_ordered) ** 2)
    plot_global = _project(global_aligned)

    kmeans_labels, _, kmeans_aligned, kmeans_cost = kmeans_procrustes_baseline(
        X,
        Y,
        n_clusters=n_pieces,
        random_state=0,
    )
    kmeans_ordered = kmeans_aligned[order]
    plot_kmeans = _project(kmeans_ordered)
    kmeans_segments = _ordered_labels(kmeans_labels[order])

    transformations, piecewise_cost, est_cutoffs = piecewise_procrustes(
        X,
        Y,
        list(order),
        k=n_pieces,
        n_samples=dp_samples,
        nuclear_norm_fn="exact",
    )
    piecewise_ordered, _ = piecewise_align(
        X,
        Y,
        list(order),
        transformations,
        est_cutoffs,
        assign_best=False,
    )
    plot_piecewise = _project(piecewise_ordered)
    piecewise_segments = _segments_from_cutoffs(est_cutoffs, len(order))

    errors = {
        "global": global_cost,
        "kmeans": kmeans_cost,
        "piecewise": piecewise_cost,
    }

    all_coords = np.vstack([plot_Y, plot_global, plot_kmeans, plot_piecewise])
    pad = 0.08
    x_min, y_min = np.min(all_coords, axis=0) - pad
    x_max, y_max = np.max(all_coords, axis=0) + pad

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12.5,
        }
    )

    fig = plt.figure(figsize=(13.0, 7.3), constrained_layout=True)
    fig.patch.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(3, 4, height_ratios=[3.4, 0.65, 0.4])
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    timeline_ax = fig.add_subplot(grid[1, :])
    score_ax = fig.add_subplot(grid[2, :])

    panels = [
        (axes[0], plot_Y, "Ground truth (Y coords)", None, None, true_cutoffs, False, true_segments, False),
        (axes[1], plot_global, "Global Procrustes (aligned X)", plot_Y, global_cost, None, True, None, True),
        (axes[2], plot_kmeans, "K-means + local rotations", plot_Y, kmeans_cost, None, True, None, True),
        (axes[3], plot_piecewise, "Piecewise DP (ours)", plot_Y, piecewise_cost, est_cutoffs, True, piecewise_segments, True),
    ]

    for ax, coords, title, reference, cost, cutoffs, show_error, segs, show_ref in panels:
        _render_panel(
            ax,
            coords,
            title=title,
            reference=reference,
            cost=cost,
            cutoffs=cutoffs,
            show_error=show_error,
            segments=segs,
            show_reference=show_ref,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    _plot_segments(
        timeline_ax,
        rows=[
            ("truth", true_segments),
            ("k-means", kmeans_segments),
            ("piecewise", piecewise_segments),
        ],
        length=len(order),
    )

    score_ax.axis("off")
    score_ax.set_facecolor(BG_COLOR)
    score_labels = [("global", "global"), ("k-means", "kmeans"), ("piecewise", "piecewise")]
    text = "  ".join(f"{label}: {errors[key]:,.1f}" for label, key in score_labels)
    score_ax.text(
        0.5,
        0.5,
        f"alignment error (lower is better) —  {text}",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#1a1a1a",
        path_effects=[patheffects.withStroke(linewidth=6, foreground=BG_COLOR)],
    )

    fig.suptitle(
        "Temporal alignment across Procrustes variants",
        fontsize=18,
        weight="bold",
        y=1.01,
        color="#111111",
    )
    fig.text(
        0.5,
        0.015,
        f"Plot shows the top two PCA components of the {dim}D trajectories; colors denote temporal segments (k={n_pieces}).",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#494949",
    )
    fig.savefig(output_path, dpi=args.dpi, facecolor=BG_COLOR)
    print(f"Saved visualization to {output_path}")

    if args.gif:
        render_alignment_gif(
            plot_X,
            plot_piecewise,
            plot_Y,
            piecewise_segments,
            est_cutoffs,
            output_path=args.gif_output,
            bounds=(x_min, x_max, y_min, y_max),
            descriptor=f"{dim} dims",
        )


if __name__ == "__main__":
    main()
