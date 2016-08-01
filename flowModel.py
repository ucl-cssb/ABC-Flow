import sys
from numpy import *
from numpy.random import *

sys.path.append("/home/cbarnes/dev/cuda-sim-area/cuda-sim/trunk/")
import cudasim
import cudasim.EulerMaruyama as EulerMaruyama
import cudasim.Gillespie as Gillespie
import cudasim.Lsoda as Lsoda

class model:
    def __init__(self, cudaFile, nspecies, nparams):
        self.cudaFile = cudaFile
        self.nspecies = nspecies
        self.nparams = nparams
        self.logp = False 
        self.modelInstance = 0


    def create_model_instance(self, beta, timepoints):
        self.beta = beta
        self.timepoints = timepoints
        self.ntimes = len(timepoints)
        
        del self.modelInstance
        self.modelInstance = Gillespie.Gillespie(self.timepoints, self.cudaFile, self.beta)


    def get_model_info(self):
        return [self.nspecies, self.nparams]

    # can handle single FP
    def convert_to_intensity(self, nFP, mu, sigma):
        signal = 0.01
        
        # Addition of normals
        if nFP > 0:
            #max: Return the maximum of an array or maximum along an axis
            #normal: Draw random samples from a normal (Gaussian) distribution.
            #signal = max(0.0001, normal(nFP*mu, sqrt(nFP*sigma*sigma)))
        	signal =  normal(nFP*mu, sqrt(nFP*sigma*sigma))
        return signal

    def simulate(self, n, pars, inits,  fps, intMus, intSgs):
        print "\tflowModel: calling cuda-sim"
        species = zeros([n, self.nspecies])
        pp = zeros([n, self.nparams])

        for i in range(n):
            species[i, :] = inits[i, :]
            
            if self.logp == False:
                pp[i, :] = pars[i, :]
            else:
                pp[i, :] = power(10, pars[i])

        simRaw = self.modelInstance.run(pp, species)

        # Fluorescence model: convert the number of FPs into intensity
        print "\tflowModel: calculating intensity"
        simInt = simRaw.astype(float)

        for i in range(n):
            for j in range(self.beta):
                for k in range(self.ntimes):
                    for l in range(len(fps)):
                        simInt[i, j, k, fps[l]] = self.convert_to_intensity(simRaw[i, j, k, fps[l]], intMus[i, l], intSgs[i, l])

        return simInt

# Return the results of cuda-sim as a list of dictionaries of the same form as we read the data
def create_dict(sims, timePoints):
    
    n = len(sims[:,0,0,0])
    ntimePoints = len(timePoints)
    #print "create_dict: ", n, timePoints, ntimePoints
    
    ret = []
    
    for j in range(n):
        retDict = {}
        for nt in range(ntimePoints):
            retDict[ timePoints[nt] ] = sims[j,:,nt,:]

        ret.append( retDict )

    return ret
