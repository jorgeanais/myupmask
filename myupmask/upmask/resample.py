"""
This module contains the function to re-sample the data given its uncertainties
"""


import numpy as np


def reSampleData(data, data_err):
    """
    Re-sample the data given its uncertainties using a normal distribution
    """

    # Gaussian random sample
    grs = np.random.normal(0., 1., data.shape[0])
    sampled_data = data + grs[:, np.newaxis] * data_err

    return sampled_data