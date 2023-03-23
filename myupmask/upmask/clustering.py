import numpy.typing as npt
from typing import Protocol


class ClusteringAlgorithm(Protocol):
    def __call__(self, data: npt.ArrayLike, cluster_size: int, **kwargs) -> npt.ArrayLike:
        ...

def kmeans(data: npt.ArrayLike, cluster_size: int, **kwargs) -> npt.ArrayLike:
    """Perform clustering using KMeans. It returns predicted cluster labels"""
    from sklearn.cluster import KMeans


    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 1000
   
    # Calculate the number of clusters
    n_clusters = int(len(data) / cluster_size)
    n_clusters = max(n_clusters, MIN_CLUSTERS)
    n_clusters = min(n_clusters, MAX_CLUSTERS)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", **kwargs)
    return kmeans.fit_predict(data)


def hdbscan(data: npt.ArrayLike, cluster_size: int, **kwargs)-> npt.ArrayLike:
    """Perform clustering using HDBSCAN. It returns predicted cluster labels"""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size)
    clusterer.fit(data)
    return clusterer.labels_