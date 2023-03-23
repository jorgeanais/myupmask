"""
Kernel o inner loop of the UPMASK algorithm.
"""

import numpy as np
import numpy.typing as npt
from typing import Protocol

from .clustering import ClusteringAlgorithm
from .spatialtest import SpatialTestRunner
from .dimred import DimensionReductionFunc


class Kernel(Protocol):
    def __call__(
            self,
            data: npt.ArrayLike,
            spatial_data: npt.ArrayLike,
            dimred_func: DimensionReductionFunc,
            clustering_algorithm: ClusteringAlgorithm,
            spatial_test_runner: SpatialTestRunner,
        ) -> npt.ArrayLike:
        ...


def default_kernel(
    data: npt.ArrayLike,
    spatial_data: npt.ArrayLike,
    dimred_func: DimensionReductionFunc,
    clustering_algorithm: ClusteringAlgorithm,
    spatial_test_runner: SpatialTestRunner,
) -> npt.ArrayLike:
    """
    Kernel function (inner loop) of the UPMASK algorithm.
    It performs the clustering and the spatial test.
    Returns the indices of the survivors
    """

    MAX_ITERATIONS_KERNEL = 25
    N_REDUCED_DIMENSIONS = 2
    CLUSTER_SIZE = 100

    # Do not modify the original arrays
    _data = data.copy()
    _spatial_data = spatial_data.copy()

    surviving_indxs = np.arange(len(_data))

    for i in range(MAX_ITERATIONS_KERNEL):
        print(f"kernel loop: {i}")
        # Perform dimensionality reduction
        _data = dimred_func(_data, n_components=N_REDUCED_DIMENSIONS)

        # Perform clustering
        cluster_labels = clustering_algorithm(_data, cluster_size=CLUSTER_SIZE)

        # Perform spatial test
        survivors = spatial_test_runner(cluster_labels, _spatial_data)

        # Exit condition (no changes after spatial test)
        if len(survivors) == len(surviving_indxs):
            print(f"Stopping kernel loop at iteration {i}.")
            break

        # Update arrays
        surviving_indxs = surviving_indxs[survivors]
        _data = _data[survivors]
        _spatial_data = _spatial_data[survivors]


    print(f"End of kernel loop. Surviving indices: {surviving_indxs}")

    return surviving_indxs
