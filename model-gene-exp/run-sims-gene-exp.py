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
matplotlib.rc('xtick', labelsize=2) 
matplotlib.rc('ytick', labelsize=2) 

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
twidth = 1.0
times = arange(0,20,twidth)
nt = len(times)

def convert_to_intensity( n_FP, mu, sigma):
    # we want multiplicative noise and log normal distributions
    #logsignal = 0
    #for i in range(n_FP):
    #    logsignal += normal(mu,sigma)
    #
    #return exp(logsignal)

    signal = 0
    for i in range(n_FP):
        signal += normal(mu,sigma)
    
    return signal

# simulate signal
def generate_signal(n,times,beta):
    ret = zeros([n, len(times)])

    parameters = zeros( [n,2] )
    species = zeros([n,1])

    for i in range(n):
        parameters[i,:] = [ 100, 0.1 ]
        species[i,:] = [ 10 ]
        #species[i,:] = [ lognormal(3.0,0.5) ]

    cudaCode = "model-gene-exp.cu"
    modelInstance = Gillespie.Gillespie(times, cudaCode, beta)
    result = modelInstance.run(parameters, species)
    print "generate_signal returned:", shape(result)

    # result is of integer type, create a new array of float type
    ret = result.astype(float)
   
    # Convert number of FPs to intensities
    # each FP contributes intensity of mu sigma
    mu = 1
    sigma = 0.01
    for i in range(n):
        for j in range(beta):
            for k in range(len(times)):
                ret[i,j,k,0] = convert_to_intensity( result[i,j,k,0], mu, sigma )

    print ret[:,:,19,0]
    
        
    return ret

def main():

    nsim = 500
    nbeta = 1

    sig = generate_signal(nsim, times, nbeta)
    print_results(sig, times, "output-data.txt")

    # histograms
    nBins = 50
    for jt in range(len(times)):
        plt.subplot(4,5,jt+1)
        plt.hist( sig[:,0,jt,0], nBins )
        plt.xlim(0,1500)
        #plt.xscale('log')
    pp.savefig()
    plt.close()

    pp.close()

main()
