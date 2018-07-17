from numpy import *
from scipy import stats
from ks_2s_ndtest import ks2d2s
from wald_wolfowitz import ww_test

def get_kd_distance1D(data, sims, ngrid, fps):
    xmin = min(data[:, 0])
    xmax = max(data[:, 0])
    xx = linspace(xmin, xmax, ngrid)
    kD = stats.gaussian_kde(data[:, 0], bw_method=0.3)
    try:
        kS = stats.gaussian_kde(sims[:, 0], bw_method=0.3)
    except:
        print "error in kd distance, all identical"
        print sims[0:5, :]
        return 1e4
    fD = reshape(kD(xx).T, xx.shape)
    fS = reshape(kS(xx).T, xx.shape)
    #d = stats.ks_2samp(fD, fS)
    #d = sum((fD-fS)*(fD-fS))
    d = stats.ks_2samp(data[:, 0], sims[:, fps])
    return d[0]

def get_kd_distance2D(data, sims, ngrid, fps):

    # KERNEL DISTANCE
    # # evaluate on the same grid as the data
    # xmin = min(data[:, 0])
    # xmax = max(data[:, 0])
    # ymin = min(data[:, 1])
    # ymax = max(data[:, 1])
    #
    # # define the grid
    # #ngrid = 10j. Complex number means stop value is inclusive
    # #mgrid returns a dense multi-dimensional meshgrid
    #
    # xx, yy = mgrid[xmin:xmax:ngrid, ymin:ymax:ngrid]
    # positions = vstack([xx.ravel(), yy.ravel()])
    #
    # # this could be run once at the start
    # vD = vstack([data[:, 0], data[:, 1]])
    # kD = stats.gaussian_kde(vD)
    # fD = reshape(kD(positions).T, xx.shape)
    #
    # vS = vstack([sims[:, 0], sims[:, 1]])
    # try:
    #     kS = stats.gaussian_kde(vS)
    # except:
    #     print "error in kd distance, all identical"
    #     print sims[0:5, :]
    #     return 1e4
    #
    # fS = reshape(kS(positions).T, xx.shape)
    #d = sum((fD-fS)*(fD-fS))
    #return da

    # 2D KS DISTANCE
    #d = ks2d2s(data[:, 0], data[:, 1], sims[:, 0], sims[:, 1], nboot=None, extra=True)
    #return abs(d[1])

    # WW distance
    #print "WW distance:", fps
    W, R = ww_test(data, sims[:,fps])
    return abs(W)

if __name__ == "__main__":
    mu1 = array([1, 10])
    sigma1 = matrix([[4, 0], [0, 1]])
    mu2 = array([1, 10])
    sigma2 = matrix([[4, 0], [0, 1]])

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    mean2 = [0+10, 0+10]
    cov2 = [[1, 0], [0, 1]]

    nres = 10
    res = zeros([nres])
    for i in range(nres):

        nsamp = 100
        data1 = random.multivariate_normal(mean, cov, nsamp)
        data2 = random.multivariate_normal(mean2, cov2, nsamp)

        d = get_kd_distance2D(data1, data2, 10j)
        res[i] = d
    print res
    print (stats.nanmean(res))


    # mu1 = array([1, 10])
    # sigma1 = matrix([[4, 0], [0, 1]])
    # mu2 = array([1, 10])
    # sigma2 = matrix([[4, 0], [0, 1]])
    #
    # nres = 10
    # res = zeros([nres])
    # for i in range(nres):
    #
    #     nsamp = 100
    #
    #     data1 = random.normal(1, 4, nsamp)
    #     data2 = random.normal(1, 4, nsamp)
    #
    #     d = get_kd_distance1D(data1, data2, 10)
    #     res[i] = d
    #
    # print mean(res)
