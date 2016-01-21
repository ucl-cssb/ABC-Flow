import sys
from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
import scipy.stats as st

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages("output-histogram.pdf")

import matplotlib
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 10}
#matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=1) 
matplotlib.rc('ytick', labelsize=1) 

sys.path.append("/home/cbarnes/dev/cuda-sim-area/cuda-sim/trunk/")
import cudasim
import cudasim.EulerMaruyama as EulerMaruyama
import cudasim.Gillespie as Gillespie
import cudasim.Lsoda as Lsoda

def print_results(result, timepoints, outfile, sx = -1, model=0):
    out = open(outfile,'w')
    print >>out, 0, 0, 0, 0,
    for i in range(len(timepoints)):
        print >>out, timepoints[i],
    print >>out, ""
    # loop over threads
    for i in range(len(result)):
        # loop over beta
        for j in range(len(result[i])):
            # loop over species
            if sx == -1:
                for l in range(len(result[i][j][0])):
                    print >>out, i,j,model,l,
                    for k in range(len(timepoints)):
                        print >>out, result[i][j][k][l],
                    print >>out, ""
            else:
                l = sx
                print >>out, i,j,model,l,
                for k in range(len(timepoints)):
                    print >>out, result[i][j][k][l],
                print >>out, ""
    out.close()

# define time structure
twidth = 0.5
times = arange(0,10,twidth)
nt = len(times)

def convert_to_intensity( n_FP, mu, sigma):
    # we want multiplicative noise and log normal distributions
    #logsignal = 0
    #for i in range(n_FP):
    #    logsignal += normal(mu,sigma)
    #
    #return exp(logsignal)

    # Add some backround
    signal = 0.01
    
    for i in range(n_FP):
        signal += normal(mu,sigma)
    
    return max(0.01, signal )


# simulate signal
def generate_signal(n,times,beta):
    ret = zeros([n, len(times)])

    parameters = zeros( [n,7] )
    species = zeros([n,2])

    for i in range(n):
        parameters[i,:] = [ 100, 2, 800, 100, 2, 800, 1 ]
        #species[i,:] = [uniform(0,5), normal(40,5)]
        species[i,:] = [10, 100]

    cudaCode = "model-gardner.cu"
    modelInstance = Gillespie.Gillespie(times, cudaCode, beta)
    result = modelInstance.run(parameters, species)
    print "generate_signal returned:", shape(result)

    # result is of integer type, create a new array of float type
    ret = result.astype(float)
   
    # Convert number of FPs to intensities
    # each FP contributes intensity of mu sigma
    mu = 1
    sigma = 5
    for i in range(n):
        for j in range(beta):
            for k in range(len(times)):
                ret[i,j,k,0] = convert_to_intensity( result[i,j,k,0], mu, sigma )
                ret[i,j,k,1] = convert_to_intensity( result[i,j,k,1], mu, sigma )

    #print ret[:,:,19,:]
    return ret

def main():

    npar = 500
    nbeta = 1

    sig = generate_signal(npar, times, nbeta)
    print_results(sig, times, "output-data.txt")

    # contour
    for jt in range(len(times)):
        #xmin, xmax = 0, 500
        #ymin, ymax = 0, 500
        #xx, yy = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        #positions = vstack([xx.ravel(), yy.ravel()])
        #values = vstack([ sig[:,0,jt,0], sig[:,0,jt,1] ])
        #kernel = st.gaussian_kde(values)
        #f = reshape(kernel(positions).T, xx.shape)
        
        #ax = plt.subplot(4,5,jt+1)
        #ax.contourf(xx, yy, f, cmap='Blues')
        #ax.set_xlim([0.1,500])
        #ax.set_ylim([0.1,500])
        #ax.set_xscale('log')
        #ax.set_yscale('log')

        xmin, xmax = -3, 3
        ymin, ymax = -3, 3
        xx, yy = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = vstack([xx.ravel(), yy.ravel()])
        values = vstack([ log10(sig[:,0,jt,0]), log10(sig[:,0,jt,1]) ])
        kernel = st.gaussian_kde(values)
        f = reshape(kernel(positions).T, xx.shape)
        
        ax = plt.subplot(4,5,jt+1)
        ax.contourf(xx, yy, f, cmap='Blues')
        ax.set_xlim([-1,3])
        ax.set_ylim([-1,3])
        #ax.set_xscale('log')
        #ax.set_yscale('log')

        
    pp.savefig()
    plt.close()
    pp.close()


    # Make some plots
    #for np in range(npar):
    #    if 0:
    #        # histograms
    #        nBins = 50
    #        for jt in range(len(times)):
    #            plt.subplot(4,5,jt+1)
    #            plt.hist( sig[np,:,jt,0], nBins )
    #            plt.xlim(0,200)
    #
    #        for jt in range(len(times)):
    #            plt.subplot(4,5,jt+10 + 1)
    #            plt.hist( sig[np,:,jt,1], nBins )
    #            plt.xlim(0,200)
    #
    #    elif 0:
    #        # scatter
    #        for jt in range(len(times)):
    #            plt.subplot(4,5,jt+1)
    #            plt.plot( sig[np,:,jt,0], sig[np,:,jt,1],'o' )
    #            plt.xlim(0,200)
    #            plt.ylim(0,200)
    #
    #    else:
    #        # contour
    #        for jt in range(len(times)):
    #
    #            if jt == 0:
    #                # causes problems at t=0
    #                continue
    #
    #            xmin, xmax = 0, 200
    #            ymin, ymax = 0, 200
    #            xx, yy = mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #            positions = vstack([xx.ravel(), yy.ravel()])
    #            values = vstack([ sig[np,:,jt,0], sig[np,:,jt,1] ])
    #            kernel = st.gaussian_kde(values)
    #            f = reshape(kernel(positions).T, xx.shape)
    #
    #            plt.subplot(4,5,jt+1)
    #            #ax = fig.gca()
    #            #ax.set_xlim(xmin, xmax)
    #            #ax.set_ylim(ymin, ymax)
    #            #cfset = ax.
    #            plt.contourf(xx, yy, f, cmap='Blues')
    #            
    #    pp.savefig()
    #    plt.close()
    #pp.close()

main()
