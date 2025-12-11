#!/usr/bin/env python3
"""Lazy Index Upgrade experiment: align Model B queries to Model A index."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import argparse
import csv
import sys
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Make project root importable so we can use the algorithms/ implementations.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.baseline_procrustes import baseline_procrustes  # type: ignore  # noqa: E402
from algorithms.piecewise_procrustes import piecewise_procrustes  # type: ignore  # noqa: E402


EMBED_DIM = 384
TOTAL_PAIRS = 10_000
TRAIN_SAMPLES = 9_000
TEST_SAMPLES = 1_000
K_CLUSTERS = 10  # for k-means
CACHE_DIR = Path("cache")

PIECEWISE_BINS = 10  # for piecewise_procrustes along norm order
PIECEWISE_SAMPLES = 2_000  # DP grid down-sampling for piecewise
PLOTS_DIR = Path("plots")

TRAIN_CSV = Path("data/quora_pairs.csv")
MODEL_TARGET_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SOURCE_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def _cache_path(model_tag: str, kind: str, count: int) -> Path:
    safe = model_tag.replace("/", "_")
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{kind}_{safe}_{count}.npy"


def load_or_encode(model: SentenceTransformer, texts: List[str], kind: str, model_tag: str) -> np.ndarray:
    cache_file = _cache_path(model_tag=model_tag, kind=kind, count=len(texts))
    if cache_file.exists():
        arr = np.load(cache_file)
        if arr.shape[0] == len(texts):
            return arr
    embeddings = model.encode(texts, show_progress_bar=True)
    arr = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_file, arr)
    return arr


def ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)


def plot_residual_hist(
    strategy: str, train_residuals: np.ndarray, test_residuals: np.ndarray
) -> Path:
    ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(train_residuals, bins=30, color="tab:blue", alpha=0.7)
    train_mean = float(np.mean(train_residuals))
    axes[0].axvline(train_mean, color="k", linestyle="--")
    axes[0].text(
        0.95,
        0.95,
        f"μ={train_mean:.4f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axes[0].set(title=f"{strategy} train residuals", xlabel="Squared error", ylabel="Frequency")

    axes[1].hist(test_residuals, bins=30, color="tab:orange", alpha=0.7)
    test_mean = float(np.mean(test_residuals))
    axes[1].axvline(test_mean, color="k", linestyle="--")
    axes[1].text(
        0.95,
        0.95,
        f"μ={test_mean:.4f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axes[1].set(title=f"{strategy} test residuals", xlabel="Squared error", ylabel="Frequency")

    plt.tight_layout()
    out_path = PLOTS_DIR / f"{strategy.lower().replace(' ', '_')}_residuals.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, eps)


class IdentityAligner:
    """No-op; represents leaving Model B queries unaligned."""

    def fit(self, *_: Sequence[np.ndarray], **__) -> "IdentityAligner":
        return self

    def transform(self, matrix: np.ndarray, **__) -> np.ndarray:
        return matrix


class GlobalAligner:
    """Single Procrustes rotation using the project implementation."""

    def __init__(self) -> None:
        self.rotation: Optional[np.ndarray] = None

    def fit(self, src: np.ndarray, tgt: np.ndarray, **__) -> "GlobalAligner":
        self.rotation = baseline_procrustes(src, tgt)
        return self

    def transform(self, matrix: np.ndarray, **__) -> np.ndarray:
        assert self.rotation is not None
        return matrix @ self.rotation


class PCAPiecewiseAligner:
    """Piecewise Procrustes ordered by the top PCA component of Model B queries."""

    def __init__(self, bins: int = PIECEWISE_BINS, n_samples: int = PIECEWISE_SAMPLES) -> None:
        self.bins = bins
        self.n_samples = n_samples
        self.transforms: Dict[int, np.ndarray] = {}
        self.bin_ranges: List[Tuple[float, float]] = []
        self.pca: Optional[PCA] = None

    def fit(self, src: np.ndarray, tgt: np.ndarray, **__ ) -> "PCAPiecewiseAligner":
        self.pca = PCA(n_components=1, svd_solver="auto", random_state=42)
        projections = self.pca.fit_transform(src).squeeze()
        order = list(np.argsort(-projections))  # sort by largest PCA coordinate
        print(
            f"[PCAPiecewise] start DP fit: bins={self.bins}, n_samples={self.n_samples}, "
            f"n_train={len(order)}"
        )
        t0 = time.time()
        transformations, _, cutoffs = piecewise_procrustes(
            src,
            tgt,
            order=order,
            k=self.bins,
            n_samples=min(len(order), self.n_samples),
        )
        print(f"[PCAPiecewise] DP fit done in {time.time() - t0:.2f}s; cutoffs={cutoffs}")

        boundaries = [0] + cutoffs + [len(order)]
        for bin_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            idxs = order[start:end]
            if len(idxs) == 0:
                self.bin_ranges.append((float("inf"), float("-inf")))
                continue
            proj_vals = projections[idxs]
            self.bin_ranges.append((float(np.min(proj_vals)), float(np.max(proj_vals))))
            if bin_id < len(transformations):
                self.transforms[bin_id] = transformations[bin_id]

        return self

    def _route(self, proj_value: float) -> int:
        for bin_id, (low, high) in enumerate(self.bin_ranges):
            if low <= proj_value <= high:
                return bin_id
        if proj_value > self.bin_ranges[0][1]:
            return 0
        return len(self.bin_ranges) - 1

    def transform(self, matrix: np.ndarray, **__) -> np.ndarray:
        assert self.pca is not None
        projections = self.pca.transform(matrix).squeeze()
        routed = np.empty_like(matrix)
        bin_ids = np.array([self._route(float(p)) for p in projections], dtype=int)
        for bin_id in np.unique(bin_ids):
            mask = bin_ids == bin_id
            rotation = self.transforms.get(bin_id)
            if rotation is None:
                routed[mask] = matrix[mask]
            else:
                routed[mask] = matrix[mask] @ rotation
        return routed


class KMeansAligner:
    def __init__(self, clusters: int = K_CLUSTERS) -> None:
        self.kmeans: Optional[KMeans] = None
        self.transforms: Dict[int, np.ndarray] = {}
        self.clusters = clusters

    def fit(self, src: np.ndarray, tgt: np.ndarray, **__) -> "KMeansAligner":
        self.kmeans = KMeans(n_clusters=self.clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(src)
        for cluster_id in range(self.clusters):
            mask = labels == cluster_id
            if np.sum(mask) == 0:
                continue
            rotation = baseline_procrustes(src[mask], tgt[mask])
            self.transforms[cluster_id] = rotation
        return self

    def transform(self, matrix: np.ndarray, **__) -> np.ndarray:
        assert self.kmeans is not None
        labels = self.kmeans.predict(matrix)
        routed = np.empty_like(matrix)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            rotation = self.transforms.get(cluster_id)
            if rotation is None:
                routed[mask] = matrix[mask]
            else:
                routed[mask] = matrix[mask] @ rotation
        return routed


def build_exact_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors.astype(np.float32))
    return index


def recall_at_k(index: faiss.IndexFlatIP, queries: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    _, neighbors = index.search(queries.astype(np.float32), k)
    hits = np.any(neighbors == targets[:, None], axis=1)
    return float(np.mean(hits))


def collect_quora_pairs(limit: int) -> Tuple[List[str], List[str]]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} must exist alongside this script.")
    queries: List[str] = []
    docs: List[str] = []
    with TRAIN_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("is_duplicate") != "1":
                continue
            q1 = (row.get("question1") or "").strip()
            q2 = (row.get("question2") or "").strip()
            if not q1 or not q2:
                continue
            queries.append(q1)
            docs.append(q2)
            if len(queries) >= limit:
                break
    if len(queries) < limit:
        raise RuntimeError(f"Only {len(queries)} duplicate pairs available in {TRAIN_CSV}.")
    return queries, docs


def evaluate_strategies(
    aligners: Sequence[Tuple[str, object]],
    index: faiss.IndexFlatIP,
    train_src: np.ndarray,
    train_tgt: np.ndarray,
    test_src: np.ndarray,
    test_tgt: np.ndarray,
    test_targets: np.ndarray,
    recall_k: int,
) -> List[dict]:
    results: List[dict] = []
    for name, aligner in aligners:
        train_aligned = aligner.transform(train_src)
        test_aligned = aligner.transform(test_src)
        train_normed = l2_normalize(train_aligned)
        test_normed = l2_normalize(test_aligned)

        train_residuals = np.sum((train_normed - train_tgt) ** 2, axis=1)
        test_residuals = np.sum((test_normed - test_tgt) ** 2, axis=1)
        recall = recall_at_k(index, test_normed, test_targets, k=recall_k)

        results.append(
            {
                "name": name,
                "recall": recall,
                "train_mean": float(np.mean(train_residuals)),
                "test_mean": float(np.mean(test_residuals)),
                "train_hist": np.histogram(train_residuals, bins=10),
                "test_hist": np.histogram(test_residuals, bins=10),
                "train_residuals": train_residuals,
                "test_residuals": test_residuals,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Lazy Index Upgrade with piecewise Procrustes.")
    parser.add_argument("--piecewise-k", type=int, default=PIECEWISE_BINS, help="Bins for piecewise Procrustes.")
    parser.add_argument(
        "--piecewise-samples",
        type=int,
        default=PIECEWISE_SAMPLES,
        help="Down-sample grid size for piecewise DP (larger is slower).",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=[1],
        help="k values for Recall@k evaluation (space-separated).",
    )
    args = parser.parse_args()
    recall_ks = args.recall_k

    print("Loading Quora duplicate pairs...")
    queries, docs = collect_quora_pairs(TOTAL_PAIRS)
    model_target = SentenceTransformer(MODEL_TARGET_NAME)
    model_source = SentenceTransformer(MODEL_SOURCE_NAME)

    print("Building frozen index with Model A documents...")
    doc_embeddings = l2_normalize(
        load_or_encode(model_target, docs, kind="docs_target", model_tag=MODEL_TARGET_NAME)
    )
    index = build_exact_index(doc_embeddings)

    print("Embedding queries with both models...")
    target_queries = l2_normalize(
        load_or_encode(model_target, queries, kind="queries_target", model_tag=MODEL_TARGET_NAME)
    )
    source_queries = load_or_encode(
        model_source, queries, kind="queries_source", model_tag=MODEL_SOURCE_NAME
    )
    source_queries = l2_normalize(source_queries)

    indices = np.arange(len(queries))
    train_idx, test_idx = train_test_split(
        indices, train_size=TRAIN_SAMPLES, test_size=TEST_SAMPLES, shuffle=False
    )

    train_src = source_queries[train_idx]
    train_tgt = target_queries[train_idx]

    test_src = source_queries[test_idx]
    test_tgt = target_queries[test_idx]

    print(f"Split {len(queries)} duplicates into {len(train_src)} train and {len(test_src)} test embeddings.")

    aligners = [
        ("Baseline", IdentityAligner()),
        ("Global", GlobalAligner().fit(train_src, train_tgt)),
        ("KMeans", KMeansAligner(clusters=args.piecewise_k).fit(train_src, train_tgt)),
        (
            "PCA Piecewise",
            PCAPiecewiseAligner(bins=args.piecewise_k, n_samples=args.piecewise_samples).fit(
                train_src, train_tgt
            ),
        ),
    ]

    test_targets = test_idx.astype(np.int64)

    print(f"Evaluating Recall@k for {len(recall_ks)} value(s)...")
    last_results: List[dict] = []
    for recall_k in recall_ks:
        print(f"\nRecall@{recall_k} results (higher is better):")
        print("Strategy         | Recall ")
        print("-----------------|--------")
        results = evaluate_strategies(
            aligners,
            index,
            train_src,
            train_tgt,
            test_src,
            test_tgt,
            test_targets,
            recall_k=recall_k,
        )
        for res in results:
            print(f"{res['name']:16} | {res['recall']:9.2%}")
        target_recall = recall_at_k(index, test_tgt, test_targets, k=recall_k)
        print(f"\nTarget (Model A) Recall@{recall_k}: {target_recall:9.2%}")
        last_results = results

    print("\nResidual statistics (MSE) per strategy:")
    for res in last_results:
        print(f"{res['name']}:")
        print(f"  Train mean residual {res['train_mean']:.6f}")
        print(f"  Test mean residual  {res['test_mean']:.6f}")
        plot_path = plot_residual_hist(
            res["name"], res["train_residuals"], res["test_residuals"]
        )
        print(f"  Saved residual histogram at {plot_path}")


if __name__ == "__main__":
    main()
