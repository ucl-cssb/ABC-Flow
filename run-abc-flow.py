# ABC FLOW ALGORITHM
import sys
from copy import deepcopy

import numpy
from numpy import *
from numpy.random import *
from scipy.stats import ks_2samp

import flowModel as model
from flowOutput import output_handler
from flowDist import get_kd_distance

class abc_flow:
    def __init__(self):
        self.data = {}
        self.nvar = 0
        self.dynPriors = 0
        self.initPriors = 0
        self.timePoints = 0
        self.ntimePoints = 0

    def read_data(self, file):
        # Reads in the data in the format:
        # time, v1, v2, v3...
        # The data is stored in a dictionary with key=timepoint, value=numpy array
        print "\n\nread_data : reading supplied file", file
        
        rawdata = genfromtxt(file, dtype=float32)
        ntot = shape(rawdata)[0]
        self.nvar = shape(rawdata)[1] - 1

        # get the number of unique time points
        self.timePoints = sort( unique( rawdata[:,0]) )
        self.ntimePoints = len(self.timePoints)
        print "read_data : number of variables:", self.nvar
        print "read_data : identified timepoints:", self.ntimePoints

        for i in range(self.ntimePoints):
            tdata = []
            for nd in range(ntot):
                if rawdata[nd,0] == self.timePoints[i]:
                    tdata.append( rawdata[nd, 1:(self.nvar+1)] )
                                         
            self.data[ self.timePoints[i] ] = array(tdata)

        for i in range(self.ntimePoints):
            print "\t", i, shape( self.data[self.timePoints[i]] )

    def set_dynamical_priors(self, dynPriorMatrix ):
        self.dynPriors = dynPriorMatrix

    def set_init_priors(self, initPriorMatrix ):
        self.initPriors = initPriorMatrix

    def set_intensity_priors(self, fps, intMeanPriorMatrix, intSigmaPriorMatrix ):
        self.fps = fps
        self.nFP = len(fps)
        self.intMeanPriors = intMeanPriorMatrix
        self.intSigmaPriors = intSigmaPriorMatrix
    
    def sample_dyn_pars(self,n):
        ret = zeros([n,self.nDynPars])
        for j in range(self.nDynPars):
            ret[:,j] = uniform( self.dynPriors[j,0], self.dynPriors[j,1], n )
        return ret

    def sample_inits(self,n):
        ret = zeros([n,self.nSpecies])
        for j in range(self.nSpecies):
            ret[:,j] = uniform( self.initPriors[j,0], self.initPriors[j,1], n )
        return ret

    def sample_int_pars(self,n):
        retMu = zeros([n,self.nFP])
        retSg = zeros([n,self.nFP])
        for j in range(self.nFP):
            retMu[:,j] = uniform( self.intMeanPriors[j,0], self.intMeanPriors[j,1], n )
            retSg[:,j] = uniform( self.intSigmaPriors[j,0], self.intSigmaPriors[j,1], n )
        return [retMu, retSg]

    # Sampling and perturbing functions
    def sample_perturb_pars(self,n,prior,prev,ids,scales):
        ret = zeros( shape(prev) )
        npar = shape(prev)[1]
        for i in range(n):
            for j in range(npar):
                ret[i,j] = self.kernel_pert(prev[ids[i],j], prior[j,:], scales[j] )
        return ret

    def compute_weights(self, n, weightsPrev, currPar, prevPar, scales):
        ret = zeros([n])
        npar = shape(currPar)[1]
        
        for i in range(n):
            for j in range(n):
                #print "\t\t:", i, j, weightsPrev[i], self.kernel_prob( npar, currPar[i,:], prevPar[j,:], scales )
                ret[i] += weightsPrev[i]*self.kernel_prob( npar, currPar[i,:], prevPar[j,:], scales )

        # Normalise
        ret = ret/float(sum(ret))
        return ret
    
    def kernel_prob(self, npar, currPar, prevPar, scales ):
        # currPar, prevPar, scales are vector valued
        for i in range(npar):
            if abs(currPar[i] - prevPar[i]) > scales[i]:
                return 0
        return 1.0
        
    def kernel_pert(self, x0, prior, scale):
        # here x0 is a scalar, prior contains the upper and lower bounds and scale is a scalar
        done = False
        while done == False:
            x = uniform(x0-scale, x0+scale)
            # print "\tx:", x0, x, scale, prior[0], prior[1]
            if x >= prior[0] and x <= prior[1]:
                done = True
        return x

    def calculate_scales(self, n, pars ):
        npar = shape(pars)[1]
        ret = zeros([npar])
        for i in range(npar):
            ret[i] = ( max(pars[:,i]) - min(pars[:,i]) )/2
        return ret

    # Simulation and comparison to data
    def compare_to_data(self, n, nbeta, sims, mode=3):
        ret = zeros([n,1])

        # create a dictionary of the same form as the data
        for j in range(n):
            sim = {}
            for nt in range(self.ntimePoints):
                sim[ self.timePoints[nt] ] = sims[j,:,nt,:]

            # Loop over dictionaries
            dist = 0
            for tp in self.timePoints:
                #print "\tcompare_to_data : data/sim shapes:", shape(self.data[tp]), shape(sim[tp])

                #if mode == 0:
                #    # One dimensional data, special case
                #    y = sorted( self.data[tp][:,0] )
                #    x = sorted( sim[tp][:,0] )
                #    if len(x) == len(y):
                #        # use the RMS
                #        dist += sum( (y-x)*(y-x) )/(self.ntimePoints*self.ntimePoints)
                #    else:
                #        print "compare_to_data : mode 1 : Difference in data and simulation dimensions"
                #        exit()

                if mode == 1:
                    # One dimensional data, general case, sum of Kolmogorov distances
                    y = sorted( self.data[tp][:,0] )
                    x = sorted( sim[tp][:,0] )

                    # KS dist
                    rr = ks_2samp(x, y)
                    dist += rr[0]

                if mode == 2:
                    # Multivariate data, use distance between kernel density estimates
                    dist += get_kd_distance(self.data[tp], sim[tp] , ngrid=10j )

            ret[j] = dist

        return ret 
            
    def check_distance(self, n, dists, eps ):
        print "\tcheck_distance : summary : ", percentile(dists,5), median(dists), percentile(dists,95)
        ret = zeros([n])
        for j in range(n):
            if dists[j] < eps:
                ret[j] = 1
            else:
                ret[j] = 0
        return ret

    
    def do_abc_rej(self, model, nbeta, nparticles, eps):
        
        nbatch = 10
        model.create_model_instance(nbeta,self.timePoints)
        self.nSpecies, self.nDynPars = model.get_model_info()
        #print "do_abc_rej : Species/Dynamical parameters in this model:", self.nSpecies, self.nDynPars

        done = False
        ntotsim = 0
        naccepted = 0
        acceptedDynParams = zeros([nparticles,self.nDynPars])
        acceptedInits = zeros([nparticles,self.nSpecies])
        acceptedIntMus = zeros([nparticles,self.nFP])
        acceptedIntSgs = zeros([nparticles,self.nFP])
        
        while done == False:
            print "\tRunning batch, nsims/nacc:", ntotsim, naccepted

            dynParameters = self.sample_dyn_pars(nbatch)
            inits = self.sample_inits(nbatch)
            intMus, intSgs = self.sample_int_pars(nbatch) 
            print "\tDone sampling"
            
            sims = model.simulate(nbatch, dynParameters, inits, self.fps, intMus, intSgs)
            print "\tDone simulation"
            dists = self.compare_to_data(nbatch, nbeta, sims, self.nFP )
            print "\tDone distance calculation"
            accMask = self.check_distance(nbatch, dists, eps)

            ntotsim += nbatch
            
            for i in range(nbatch):
                if accMask[i] == 1 and naccepted < nparticles:
                    acceptedDynParams[naccepted,:] = dynParameters[i,:]
                    acceptedInits[naccepted,:] = inits[i,:]
                    acceptedIntMus[naccepted,:] = intMus[i,]
                    acceptedIntSgs[naccepted,:] = intSgs[i,]

                    naccepted += 1
                if naccepted == nparticles:
                    done = True

        print "do_abc_rej : Completed"
        print "           : Final acceptance rate = ", naccepted/float(ntotsim)
        return [acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs]

    def do_abc_smc(self, model, nbeta, nparticles, epsilons):

        npop = len(epsilons)
        nbatch = 10
        model.create_model_instance(nbeta,self.timePoints)
        self.nSpecies, self.nDynPars = model.get_model_info()

        # do the first population sampling from the prior
        print "do_abc_smc : Population 0"
        acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs = self.do_abc_rej(model, nbeta, nparticles, epsilons[0])
        acceptedWeights = ones([nparticles])/float(nparticles)

        for pop in range(1,npop):
            currDynParams = zeros([nparticles,self.nDynPars])
            currInits = zeros([nparticles,self.nSpecies])
            currIntMus = zeros([nparticles,self.nFP])
            currIntSgs = zeros([nparticles,self.nFP])

            # calculate the scales for this population
            scales_dyn = self.calculate_scales(nparticles, acceptedDynParams)
            scales_inits = self.calculate_scales(nparticles, acceptedInits)
            scales_Mus = self.calculate_scales(nparticles, acceptedIntMus)
            scales_Sgs =  self.calculate_scales(nparticles, acceptedIntSgs)

            # Rejection stage
            doneRej = False
            ntotsim = 0
            naccepted = 0
            while doneRej == False:
                print "\tRunning batch, nsims/nacc:", ntotsim, naccepted

                ids = numpy.random.choice( nparticles, size=nbatch, replace=True, p=acceptedWeights)
                print "sampled ids:", ids
                dynParameters = self.sample_perturb_pars(nbatch, self.dynPriors, acceptedDynParams, ids, scales=scales_dyn )
                inits         = self.sample_perturb_pars(nbatch, self.initPriors, acceptedInits, ids, scales=scales_inits)
                intMus        = self.sample_perturb_pars(nbatch, self.intMeanPriors, acceptedIntMus, ids, scales=scales_Mus)
                intSgs        = self.sample_perturb_pars(nbatch, self.intSigmaPriors, acceptedIntSgs, ids, scales=scales_Sgs) 
                print "\tDone sampling"

                sims = model.simulate(nbatch, dynParameters, inits, self.fps, intMus, intSgs)
                dists = self.compare_to_data(nbatch, nbeta, sims, self.nFP )
                accMask = self.check_distance(nbatch, dists, epsilons[pop])

                ntotsim += nbatch

                for i in range(nbatch):
                    if accMask[i] == 1 and naccepted < nparticles:
                        currDynParams[naccepted,:] = dynParameters[i,:]
                        currInits[naccepted,:] = inits[i,:]
                        currIntMus[naccepted,:] = intMus[i,]
                        currIntSgs[naccepted,:] = intSgs[i,]

                        naccepted += 1
                    if naccepted == nparticles:
                        doneRej = True

            print "do_abc_smc : Population", pop, "\tacceptance rate = ", naccepted/float(ntotsim)

            # update weights
            currPar = column_stack( (currDynParams, currInits, currIntMus, currIntSgs) )
            prevPar = column_stack( (acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs) )
            all_scales = concatenate( (scales_dyn, scales_inits, scales_Mus, scales_Sgs) )
            # print shape(currDynParams), shape(currInits), shape(currIntMus), shape(currIntSgs)
            # print shape(currPar), shape(prevPar)
            acceptedWeights = self.compute_weights(nparticles, acceptedWeights, currPar, prevPar, scales=all_scales )
            print "acceptedWeights:"
            print acceptedWeights
            
            # update best estimates
            acceptedDynParams = deepcopy( currDynParams )
            acceptedInits = deepcopy( currInits )
            acceptedIntMus = deepcopy( currIntMus )
            acceptedIntSgs = deepcopy( currIntSgs )

        print "do_abc_smc : Completed successfully"
        return [acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs, acceptedWeights]

def main():

    abcAlg = abc_flow()


    if 0:
        abcAlg.read_data("model-gene-exp/flow-data-gene-exp.txt")

        # plot the data
        outHan = output_handler()
        outHan.plot_data_dict_1D("plot-gene-exp-data.pdf",abcAlg.data,abcAlg.timePoints)

        # define the model
        immdeath = model.model("model-gene-exp/model-gene-exp.cu", nspecies=1, nparams=2)

        # priors assumed uniform, matrix of size npar x 2 (min max)
        dynPriorMatrix = array( [[90, 110], [0.09, 0.11]] )
        #dynPriorMatrix = array( [[0.0, 200], [0.0, 0.2]] )
        initPriorMatrix = array( [[5, 15]] )

        # Set which fluorescent proteins are measured
        # and their means and sigmas
        nfp = 1
        fps = [ 0 ]
        intMeanPriorMatrix = array( [[0.8, 1.2]])
        intSigmaPriorMatrix = array( [[0.008, 0.012]] )

        # Set the internal variables
        abcAlg.set_dynamical_priors( dynPriorMatrix )
        abcAlg.set_init_priors( initPriorMatrix )
        abcAlg.set_intensity_priors( fps, intMeanPriorMatrix, intSigmaPriorMatrix )

        # ABC algorithm
        epsilon = 5
        nparticles = 10
        nbeta = 500
        accPars, accInit, accMus, accSgs = abcAlg.do_abc_rej(immdeath, nbeta, nparticles, epsilon)

        # calculate posterior medians
        medPars = zeros([1,immdeath.nparams])
        medInit = zeros([1,immdeath.nspecies])
        medMu = zeros([1,nfp])
        medSg = zeros([1,nfp])
        
        medPars[0,:] = [median(accPars[:,i]) for i in range(immdeath.nparams)]
        medInit[0,:] = [median(accInit[:,i]) for i in range(immdeath.nspecies)]
        medMu[0,:] = [median(accMus[:,i]) for i in range(nfp)]
        medSg[0,:] = [median(accSgs[:,i]) for i in range(nfp)]
        print "posterior median value dynpar/inits :", medPars, medInit, medMu, medSg

        outHan.make_post_hists("plot-gene-exp-posteriors-dyn.pdf", accPars, [0,1] )
        outHan.make_post_hists("plot-gene-exp-posteriors-init.pdf", accInit, [0] )
        outHan.make_post_hists("plot-gene-exp-posteriors-mu.pdf", accMus, [0] )
        outHan.make_post_hists("plot-gene-exp-posteriors-sg.pdf", accSgs, [0] )

        # resimulate the model with the posterior medians
        immdeath.create_model_instance(nbeta,abcAlg.timePoints)
        res = immdeath.simulate(1, medPars, medInit, fps, medMu, medSg )

        # convert the output of cuda-sim into a data dictionary
        resDict = model.create_dict(res,abcAlg.timePoints)[0]
        # make some plots
        outHan.make_qq_plots("plot-gene-exp-final-qq.pdf", abcAlg.data, resDict, abcAlg.timePoints)
        outHan.make_comp_plot_1D("plot-gene-exp-final-fit.pdf", abcAlg.data, resDict, abcAlg.timePoints)

    if 0:
        abcAlg.read_data("model-gardner/flow-data-gardner.txt")

        # plot the data
        outHan = output_handler()
        outHan.plot_data_dict_2D("plot-gardner-data.pdf",abcAlg.data,abcAlg.timePoints)

        # define the model
        gardner = model.model("model-gardner/model-gardner.cu", nspecies=2, nparams=7)

        # priors
        # real values : 100, 2, 800, 100, 2, 800, 1
        dynPriorMatrix = array( [ [80, 120],[1,3],[700,900], [80, 120],[1,3],[700,900], [0.5,1.5] ] )
        initPriorMatrix = array( [[0, 20], [50,150]] )

        # Set which fluorescent proteins are measured and their means and sigmas
        nfp = 1
        fps = [0]
        # real values : 1, 5
        intMeanPriorMatrix = array( [[0.9, 1.1] ])
        intSigmaPriorMatrix = array( [[4.5, 5.5] ])

        # Set the internal variables
        abcAlg.set_dynamical_priors( dynPriorMatrix )
        abcAlg.set_init_priors( initPriorMatrix )
        abcAlg.set_intensity_priors( fps, intMeanPriorMatrix, intSigmaPriorMatrix )

        epsilon = 5
        nparticles = 100
        nbeta = 500
        accPars, accInit, accMus, accSgs = abcAlg.do_abc_rej(gardner, nbeta, nparticles, epsilon)

        # calculate posterior medians
        medPars = zeros([1,gardner.nparams])
        medInit = zeros([1,gardner.nspecies])
        medMu = zeros([1,nfp])
        medSg = zeros([1,nfp])
        
        medPars[0,:] = [median(accPars[:,i]) for i in range(gardner.nparams)]
        medInit[0,:] = [median(accInit[:,i]) for i in range(gardner.nspecies)]
        medMu[0,:] = [median(accMus[:,i]) for i in range(nfp)]
        medSg[0,:] = [median(accSgs[:,i]) for i in range(nfp)]
        print "posterior median valuse dynpar/inits :", medPars, medInit, medMu, medSg

        outHan.make_post_hists("plot-gardner-posteriors-dyn.pdf", accPars, [0,1,2,3,4,5,6] )
        outHan.make_post_hists("plot-gardner-posteriors-init.pdf", accInit, [0,1] )
        outHan.make_post_hists("plot-gardner-posteriors-mu.pdf", accMus, [0] )
        outHan.make_post_hists("plot-gardner-posteriors-sg.pdf", accSgs, [0] )

    if 1:
        abcAlg.read_data("model-gardner/flow-data-gardner.txt")

        # plot the data
        outHan = output_handler()
        outHan.plot_data_dict_2D("plot-gardner-data.pdf",abcAlg.data,abcAlg.timePoints)

        # define the model
        gardner = model.model("model-gardner/model-gardner.cu", nspecies=2, nparams=7)

        # priors
        # real values : 100, 2, 800, 100, 2, 800, 1
        dynPriorMatrix = array( [ [80, 120],[1,3],[700,900], [80, 120],[1,3],[700,900], [0.5,1.5] ] )
        initPriorMatrix = array( [[0, 20], [50,150]] )

        # Set which fluorescent proteins are measured
        # and their means and sigmas
        nfp = 2
        fps = [ 0, 1 ]
        # real values : 1, 5
        intMeanPriorMatrix = array( [[0.9, 1.1], [0.9, 1.1] ])
        intSigmaPriorMatrix = array( [[4.5, 5.5], [4.5, 5.5] ] )

        # Set the internal variables
        abcAlg.set_dynamical_priors( dynPriorMatrix )
        abcAlg.set_init_priors( initPriorMatrix )
        abcAlg.set_intensity_priors( fps, intMeanPriorMatrix, intSigmaPriorMatrix )

        epsilons = [0.001, 0.0005, 0.00001]
        nparticles = 100
        nbeta = 500
        accPars, accInit, accMus, accSgs, accWeights = abcAlg.do_abc_smc(gardner, nbeta, nparticles, epsilons)

        # calculate posterior medians
        medPars = zeros([1,gardner.nparams])
        medInit = zeros([1,gardner.nspecies])
        medMu = zeros([1,nfp])
        medSg = zeros([1,nfp])
        
        medPars[0,:] = [median(accPars[:,i]) for i in range(gardner.nparams)]
        medInit[0,:] = [median(accInit[:,i]) for i in range(gardner.nspecies)]
        medMu[0,:] = [median(accMus[:,i]) for i in range(nfp)]
        medSg[0,:] = [median(accSgs[:,i]) for i in range(nfp)]
        print "posterior median valuse dynpar/inits :", medPars, medInit, medMu, medSg

        outHan.make_post_hists("plot-gardner-2D-posteriors-dyn.pdf", accPars, [0,1,2,3,4,5,6] )
        outHan.make_post_hists("plot-gardner-2D-posteriors-init.pdf", accInit, [0,1] )
        outHan.make_post_hists("plot-gardner-2D-posteriors-mu.pdf", accMus, [0] )
        outHan.make_post_hists("plot-gardner-2D-posteriors-sg.pdf", accSgs, [0] )
main()
