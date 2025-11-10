"""Procrustes alignment experiments.

This module generates paired embeddings for a set of sample sentences using a
public Hugging Face model, runs orthogonal Procrustes alignment, visualises the
residual error, and then fits multiple orthogonal transformations by
clustering.
"""

from __future__ import annotations

import dataclasses
import importlib
import itertools
from typing import List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class HuggingFaceSentenceEncoder:
    """Encode sentences with a Hugging Face model using mean pooling."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._embedding_dim: int | None = None

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise ValueError("Call `encode` before accessing `embedding_dim`.")
        return self._embedding_dim

    def encode(self, sentences: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return sentence embeddings as NumPy arrays."""

        outputs: List[np.ndarray] = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                model_output = self.model(**inputs)

            hidden = model_output.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            masked_hidden = hidden * mask
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)
            pooled = masked_hidden.sum(dim=1) / mask_sum
            outputs.append(pooled.cpu().numpy())

            if self._embedding_dim is None:
                self._embedding_dim = pooled.shape[1]

        if not outputs:
            raise ValueError("No sentences provided for encoding.")

        return np.vstack(outputs)


def generate_embeddings(
    sentences: Sequence[str],
    encoder: HuggingFaceSentenceEncoder,
    k_groups: int,
    x_dim: int | None = None,
    y_dim: int | None = None,
    noise: float = 0.05,
    seed: int = 13,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired embeddings ``X`` and ``Y`` with optional partitions.

    ``Y`` is created by applying one of ``k_groups`` orthogonal transformations
    (plus small Gaussian noise) to the base embedding.  The returned ``labels``
    correspond to the transformation used for each sentence.
    """

    if not sentences:
        raise ValueError("Expected at least one sentence to encode.")

    base_embeddings = encoder.encode(sentences)
    base_dim = base_embeddings.shape[1]

    if x_dim is None:
        x_dim = base_dim
    if y_dim is None:
        y_dim = base_dim
    if x_dim > base_dim or y_dim > base_dim:
        raise ValueError(
            "Requested projection dimensions exceed the encoder's embedding width."
        )

    rng = np.random.default_rng(seed)

    # Project into the two requested spaces.
    X = base_embeddings[:, :x_dim]

    labels = rng.integers(0, k_groups, size=len(sentences))
    transformations = []
    for group in range(k_groups):
        random_matrix = rng.normal(size=(base_dim, base_dim))
        u, _, vt = np.linalg.svd(random_matrix, full_matrices=False)
        transformations.append(u @ vt)

    padded_Y = np.zeros((len(sentences), base_dim))
    for idx, (embedding, label) in enumerate(zip(base_embeddings, labels)):
        transformed = embedding @ transformations[label]
        noise_vec = rng.normal(scale=noise, size=base_dim)
        padded_Y[idx] = transformed + noise_vec

    Y = padded_Y[:, :y_dim]
    return X, Y, labels


def pad_embeddings(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad two embedding matrices with zeros so they have matching width."""

    x_dim = X.shape[1]
    y_dim = Y.shape[1]
    if x_dim == y_dim:
        return X.copy(), Y.copy()

    target = max(x_dim, y_dim)
    padded_X = np.zeros((X.shape[0], target))
    padded_Y = np.zeros((Y.shape[0], target))
    padded_X[:, :x_dim] = X
    padded_Y[:, :y_dim] = Y
    return padded_X, padded_Y


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return the orthogonal matrix that aligns ``X`` to ``Y``."""

    cross_cov = X.T @ Y
    U, _, Vt = np.linalg.svd(cross_cov, full_matrices=False)
    return U @ Vt


def residuals(X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return the per-row residual norm after applying ``Q`` to ``X``."""

    return np.linalg.norm(X @ Q - Y, axis=1)


@dataclasses.dataclass
class ClusterResult:
    transformations: List[np.ndarray]
    assignments: np.ndarray
    errors: np.ndarray


def clustered_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    max_iter: int = 50,
    seed: int = 17,
) -> ClusterResult:
    """Cluster points and fit one orthogonal map per cluster.

    This is a simple hard-EM style procedure: alternate between estimating
    the orthogonal transformations and reassigning each example to the cluster
    that minimises its alignment error.
    """

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    assignments = rng.integers(0, k, size=n)

    transformations = [np.eye(X.shape[1]) for _ in range(k)]
    errors = np.zeros((n, k))

    for iteration in range(max_iter):
        # Step 1: fit transformations given the assignments.
        for cluster in range(k):
            members = np.where(assignments == cluster)[0]
            if len(members) == 0:
                # Re-seed the empty cluster with the point of largest error.
                farthest = np.argmax(errors.max(axis=1)) if iteration > 0 else rng.integers(0, n)
                assignments[farthest] = cluster
                members = np.array([farthest])

            Xc = X[members]
            Yc = Y[members]
            if len(members) == 1:
                transformations[cluster] = np.eye(X.shape[1])
            else:
                transformations[cluster] = orthogonal_procrustes(Xc, Yc)

        # Step 2: reassign points to the best fitting transformation.
        for cluster, Q in enumerate(transformations):
            errors[:, cluster] = residuals(X, Y, Q)

        new_assignments = np.argmin(errors, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

    final_errors = errors[np.arange(n), assignments]
    return ClusterResult(transformations, assignments, final_errors)


def best_label_agreement(true_labels: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Return the best possible label agreement after permuting cluster ids."""

    best = 0.0
    for perm in itertools.permutations(range(k)):
        mapping = np.array(perm)
        remapped = mapping[predicted]
        score = np.mean(remapped == true_labels)
        if score > best:
            best = score
    return best


def plot_residual_hist(residual_values: np.ndarray, title: str, path: str) -> None:
    """Plot (or log) a histogram of the residual values.

    ``matplotlib`` is used when available; otherwise, an ASCII histogram is
    written to ``path`` with a ``.txt`` suffix.
    """

    if importlib.util.find_spec("matplotlib.pyplot") is None:
        counts, bin_edges = np.histogram(residual_values, bins=20)
        lines = [title, "Bin start, Bin end, Count"]
        for start, end, count in zip(bin_edges[:-1], bin_edges[1:], counts):
            lines.append(f"{start:.4f}, {end:.4f}, {int(count)}")
        fallback_path = path.rsplit(".", 1)[0] + ".txt"
        with open(fallback_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"matplotlib unavailable; wrote histogram data to {fallback_path}")
        return

    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(8, 4))
    plt.hist(residual_values, bins=20, color="steelblue", edgecolor="white")
    plt.title(title)
    plt.xlabel("Residual norm")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    sentences = [
        "how to make sourdough starter",
        "weather tomorrow in new york",
        "what is the capital of france",
        "train schedule from boston to chicago",
        "best hiking trails near seattle",
        "python list comprehension examples",
        "symptoms of seasonal allergies",
        "fix slow laptop performance",
        "restaurants open late near me",
        "learn guitar chords fast",
        "history of the silk road",
        "mars rover latest news",
        "tips for public speaking",
        "cheap flights to tokyo",
        "top rated sci fi books",
        "healthy vegetarian recipes",
        "install solar panels home",
        "machine learning tutorials",
        "indie games to watch",
        "plan weekend road trip",
    ]

    encoder = HuggingFaceSentenceEncoder()
    X_raw, Y_raw, true_labels = generate_embeddings(
        sentences,
        encoder=encoder,
        x_dim=64,
        y_dim=72,
        k_groups=3,
        noise=0.03,
    )

    print(
        f"Using model '{encoder.model_name}' with embedding dimension {encoder.embedding_dim}"
    )

    X, Y = pad_embeddings(X_raw, Y_raw)

    Q = orthogonal_procrustes(X, Y)
    single_residuals = residuals(X, Y, Q)
    print(f"Average residual (single transform): {single_residuals.mean():.4f}")

    plot_residual_hist(single_residuals, "Residuals with single Procrustes map", "residual_histogram.png")
    print("Saved residual histogram to residual_histogram.png")

    clustered = clustered_procrustes(X, Y, k=3)
    print("\nClustered Procrustes summary:")
    for cluster_id in range(3):
        cluster_mask = clustered.assignments == cluster_id
        count = cluster_mask.sum()
        if count == 0:
            print(f"  Cluster {cluster_id}: empty")
            continue
        cluster_error = clustered.errors[cluster_mask]
        print(
            f"  Cluster {cluster_id}: size={count}, avg_residual={cluster_error.mean():.4f}, "
            f"max_residual={cluster_error.max():.4f}"
        )

    agreement = best_label_agreement(true_labels, clustered.assignments, k=3)
    print(f"Label agreement with ground truth partitions: {agreement:.2%}")

    plot_residual_hist(
        clustered.errors,
        "Residuals with clustered Procrustes maps",
        "clustered_residual_histogram.png",
    )
    print("Saved clustered residual histogram to clustered_residual_histogram.png")


if __name__ == "__main__":
    main()
