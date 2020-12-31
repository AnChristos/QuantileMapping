import numpy as np


def npCI(data,
         q,
         sigma=1.96,
         assume_sorted=False):
    ''' Implements non-parametric intervals for quantiles following
        "Confidence interval for quantiles and percentiles
         Cristiano Ialongo"

        data : array of observasions
        q: Percentile or sequence of percentiles
        to compute uncertainties for
        assume_sorted = False : If True the data are assumed
        to be sorted
        sigma = 1.96 : standarized quantile Z_(a/2)
        for confidence level

        return lower,upper arrays

        Note : Can be inaccurate for extreme quantiles
    '''
    if assume_sorted:
        _data = data
    else:
        _data = np.sort(data)

    _n = len(_data)
    _l_indices = np.floor(
        (_n*q) - sigma * np.power(_n*q*(1.-q), 0.5)).astype('int') - 1
    _u_indices = np.ceil(
        (_n*q) + sigma * np.power(_n*q*(1.-q), 0.5)).astype('int') - 1
    _l_indices = np.clip(_l_indices, 0, len(_data)-1)
    _u_indices = np.clip(_u_indices, 0, len(_data)-1)
    return data[_l_indices], data[_u_indices]
