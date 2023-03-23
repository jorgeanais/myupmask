"""
Module used to test if the data is spatially clustered
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt



class SpatialTest(Protocol):
    def __call__(self, data: npt.ArrayLike) -> bool:
        ...

class SpatialTestRunner(Protocol):
    def __call__(
        self,
        labels: npt.ArrayLike,
        spatial_data: npt.ArrayLike,
        spatial_test: SpatialTest
    ) -> npt.ArrayLike:
        ...


def ripley_k_func(data: npt.ArrayLike) -> bool:
    """
    Random field rejection method using Ripley's K function.
    Assumed that the data is a rectangular area between 0 and 1.
    Return true if the data is not similar to a uniform random
    distribution.

    Based on pyUPMASK implementation.

    Notes:
      - RADII is restricted to values up to 1/4 of the smaller side of a
        rectangle (Ripley, 1977, 1988; Diggle, 1983)
      - CRITICAL_FACTOR in C_thresh correspond to a 1% critical value.
        From Dixon (2001), 'Ripley's K function'.
    """

    from astropy.stats import RipleysKEstimator

    AREA = 1
    XMIN = 0
    XMAX = 1
    YMIN = 0
    YMAX = 1
    CRITICAL_FACTOR = 1.68
    RADII = np.linspace(0.01, 0.25, 50)
    MAX_SOURCES = 5000  # Max number of

    n_sources = len(data)

    mode = "translation"  # if n_sources > MAX_SOURCES else "none"
    k_est = RipleysKEstimator(area=AREA, x_max=XMAX, y_max=YMAX, x_min=XMIN, y_min=YMIN)
    L_t = k_est.Lfunction(data, RADII, mode=mode)

    C_s = -np.inf if np.isnan(L_t).all() else np.nanmax(abs(L_t - RADII))
    C_thresh = CRITICAL_FACTOR / n_sources
    result = C_s >= C_thresh

    return result


def sp_test_runner(
    labels: npt.ArrayLike,
    spatial_data: npt.ArrayLike,
    spatial_test: SpatialTest = ripley_k_func,
) -> npt.ArrayLike:
    """
    Check if the data is compatible with given spatial distribution test.
    Each group is tested individually using their clustering labels.
    Negative labels are considered noise, and excluded from the process.
    Return the index of the surviving points.
    """

    MIN_MEMBER_SIZE = 3

    if len(labels) != len(spatial_data):
        raise ValueError("Spatial data and labels must have the same length")

    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Remove groups with less than MIN_MEMBER_SIZE
    unique_labels = unique_labels[counts >= MIN_MEMBER_SIZE]

    # Exclude groups with negative labels (noise)
    unique_labels = unique_labels[unique_labels >= 0]  

    if len(unique_labels) < 1:
        raise ValueError("No groups found during spatial test")

    surviving_indxs = np.array([], dtype=int)

    for label in unique_labels:

        members_indxs = np.where(labels == label)
        group_data = spatial_data[members_indxs]

        if spatial_test(group_data):
            surviving_indxs = np.append(surviving_indxs, members_indxs)

    return surviving_indxs
