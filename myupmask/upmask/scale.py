"""
This module contain all the functions to scale the data
"""

import numpy.typing as npt
from sklearn.preprocessing import StandardScaler


def std_scale(data: npt.ArrayLike):
    """Scale the data to unit variance and zero mean"""
    return StandardScaler().fit(data).transform(data)