from numpy import *
from scipy import stats

def get_kd_distance(data, sims, ngrid):
    # evaluate on the same grid as the data
    xmin = min(data[:,0])
    xmax = max(data[:,0])
    ymin = min(data[:,1])
    ymax = max(data[:,1])

    # define the grid
    #ngrid = 10j. Complex number means stop value is inclusive
    #mgrid returns a dense multi-dimensional meshgrid

    xx, yy = mgrid[xmin:xmax:ngrid, ymin:ymax:ngrid ]
    positions = vstack([xx.ravel(), yy.ravel()])

    # this could be run once at the start
    vD = vstack([ data[:,0], data[:,1] ])
    kD = stats.gaussian_kde(vD)
    fD = reshape(kD(positions).T, xx.shape)

    vS = vstack([ sims[:,0], sims[:,1] ])
    try:
        kS = stats.gaussian_kde(vS)
    except:
        print "error in kd distance, all identical"
        print sims[0:5,:]
        return 1e6

    fS = reshape(kS(positions).T, xx.shape)

    d = sum( (fD-fS)*(fD-fS) )
    #print "\t\tkd distance:", d

    return d


if __name__ == "__main__":
    mu1=array([1,10])
    sigma1=matrix([ [4,0],[0,1] ])
    mu2=array([1,10])
    sigma2=matrix([ [4,0],[0,1] ])

    nres = 10
    res = zeros([nres])
    for i in range(nres):

        nsamp = 100
        data1 = random.multivariate_normal(mu1,sigma1,nsamp)
        data2 = random.multivariate_normal(mu2,sigma2,nsamp)

        d = get_kd_distance(data1,data2)
        res[i] = d

    print mean(res)
