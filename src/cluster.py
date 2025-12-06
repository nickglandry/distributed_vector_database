import numpy as np
from sklearn.cluster import KMeans
from typing import List, Sequence


def compute_centroids(vectors: Sequence[Sequence[float]], num_clusters: int) -> List[List[float]]:
    """
    Compute K centroids from a list of vectors using sklearn KMeans.

    Args:
        vectors: List (or sequence) of vectors, each a list of floats.
        num_clusters: Number of centroids to compute (your number of shards).

    Returns:
        A list of centroids, each centroid being a list of floats.
    """

    # Convert to numpy array
    X = np.array(vectors, dtype=float)

    # Ensure we have enough samples
    if len(X) < num_clusters:
        raise ValueError("Number of vectors must be >= number of clusters")

    # Run KMeans
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init="auto",
        random_state=42
    )
    kmeans.fit(X)

    # Return centroids as plain Python lists
    return kmeans.cluster_centers_.tolist()
