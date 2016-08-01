from __future__ import division
import numpy as np
from numpy import random
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

__all__ = ['ks2d2s', 'estat']

def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples.
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.
    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)
    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''

    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    sqen = np.sqrt(n1*n2/(n1+n2))

    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    D = (D1 + D2)/2

    r1 = pearsonr(x1, y1)[0]
    r2 = pearsonr(x2, y2)[0]

    if np.isfinite(r2) == False:
        r2 = 0
    r = np.sqrt(1 - (r1**2 + r2**2)/2)

    if nboot is None:
        D = D * sqen/(1 + r*(0.25 - 0.75/sqen))
        #p = kstwobign.sf(D)
        p = 0
    else:
        raise NotImplementedError
    if extra:
        return p, D
    else:
        return p


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.zeros((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1-a2, b1-b2, c1-c2, d1-d2]
    D1[:,0] -= 1/n1         # re-assign the point to maximize difference, the
    #D1[D1 >= 0] += 1/n1     # discrepancy is significant for N < ~50
    #D1 = np.abs(D1).max()
    ix = np.unravel_index(np.argmax(np.abs(D1)), D1.shape)
    D1 = D1[ix]
    if D1 >=0:
        D1 += 1/n1
    return D1


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum( ix1 &  ix2)/n
    b = np.sum( ix1 & ~ix2)/n
    c = np.sum(~ix1 &  ix2)/n
    d = 1 - a - b -c
    return a, b, c, d

