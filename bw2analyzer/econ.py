from __future__ import division
import numpy as np


def gini_coefficient(x):
    """
    Return computed Gini coefficient.

    See http://en.wikipedia.org/wiki/Gini_coefficient

    Adapted from econpy library.
    copyright: 2005-2009 Alan G. Isaac
    license: MIT license
    contact: aisaac AT american.edu

    Args:
        *x* (list or array): Data

    Returns:
        Gini coefficient (float)

    """
    x = np.array(x)
    x.sort()
    y = np.cumsum(x)
    length = float(x.shape[0])
    B = y.sum() / (y[-1] * length)
    return float(1. + 1 / length - 2 * B)


def herfindahl_index(x, normalize=True):
    """
    Return computed Herfindahl index.

    See http://en.wikipedia.org/wiki/Herfindahl_index

    Normalized scores are bounded [0, 1]; non-normalized scores are [1/len(x), 1]. Normalization only counts non-zero values.

    Args:
        *x* (list or array): Data
        *normalize* (bool, default=True): Flag to normalize scores.

    Returns:
        Herfindahl index (float)

    """
    # Normalize so that total is 1
    print x
    x = np.array(x) / np.sum(x)
    print x
    index = (x ** 2).sum()
    if normalize:
        correction = 1 / (x != 0).sum()
        print index, correction
        index = (index - correction) / (1 - correction)
    return float(index)


def concentration_ratio(x, number=4):
    """
    Return computed concentration ratio.

    See http://en.wikipedia.org/wiki/Concentration_ratio

    The concentration ratio measures the share of the market controlled by the top *number* firms. Returned ratio values vary from 0 to 1.

    Args:
        *x* (list or array): Data
        *number* (int, default=4): Number of values to consider. 4 and 8 are commonly used.

    Returns:
        Concentration ratio (float)

    """
    # Normalize so that total is 1
    x = np.array(x) / np.sum(x)
    x.sort()
    return float(x[-number:].sum())

