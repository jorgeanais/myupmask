"""
Module for the outer loop of the UPMASK algorithm.
"""

from functools import partial
import numpy as np
import numpy.typing as npt

from .kernel import Kernel


def outer_loop(
    data: npt.ArrayLike,
    spatial_data: npt.ArrayLike,
    kernel: Kernel,
    kernel_params: dict,
):
    """
    Outer loop of the UPMASK algorithm.
    Returns the probability of each data point to be selected.
    """

    MAX_ITERATIONS_OUTER = 25

    # Add data and spatial data to the kernel parameters
    kernel_params |= {"data": data, "spatial_data": spatial_data}

    surviving_indxs = np.array([], dtype=int)

    for i in range(MAX_ITERATIONS_OUTER):
        print(f"outer loop iteration: {i}")
        temp_surviving = partial(kernel, **kernel_params)()
        surviving_indxs = np.concatenate((surviving_indxs, temp_surviving))
    
    indxs, counts = np.unique(surviving_indxs, return_counts=True)
    probs = counts / MAX_ITERATIONS_OUTER
    results = np.zeros(len(data))
    results[indxs] = probs
    return results




