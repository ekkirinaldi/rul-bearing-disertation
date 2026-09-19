"""SAE latent visualization: UMAP 2D embedding + HDBSCAN clustering, colored by phase."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reduce_umap(
    latents: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is required: pip install umap-learn") from e
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(latents)


def cluster_hdbscan(
    embedding: np.ndarray,
    *,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError("hdbscan is required: pip install hdbscan") from e
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(embedding)


def plot_latent_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    *,
    title: str = "SAE latent UMAP",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=4, cmap="tab10", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.colorbar(sc, ax=ax, fraction=0.04)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def assign_phases(rul: np.ndarray, healthy_threshold: float = 0.7, prefailure_threshold: float = 0.2) -> np.ndarray:
    """Assign 0=healthy, 1=wear, 2=pre-failure based on RUL value."""
    out = np.empty(rul.shape, dtype=np.int32)
    out[rul >= healthy_threshold] = 0
    out[(rul < healthy_threshold) & (rul >= prefailure_threshold)] = 1
    out[rul < prefailure_threshold] = 2
    return out
