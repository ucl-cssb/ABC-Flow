# ABC FLOW ALGORITHM
from copy import deepcopy
import numpy
from numpy import *
from numpy.random import *
from xml.etree import ElementTree
import sys, getopt
import os
import flowModel as model
from flowOutput import output_handler
from flowDist import get_kd_distance1D
from flowDist import get_kd_distance2D

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
        self.timePoints = sort(unique(rawdata[:, 0]))
        self.ntimePoints = len(self.timePoints)
        print "read_data : number of variables:", self.nvar
        print "read_data : identified timepoints:", self.ntimePoints

        for i in range(self.ntimePoints):
            tdata = []
            for nd in range(ntot):
                if rawdata[nd, 0] == self.timePoints[i]:
                    tdata.append(rawdata[nd, 1:(self.nvar+1)])
                                         
            self.data[ self.timePoints[i]] = array(tdata)

        for i in range(self.ntimePoints):
            print "\t", i, shape(self.data[self.timePoints[i]])

    def set_dynamical_priors(self, dynPriorMatrix):
        self.dynPriors = dynPriorMatrix

    def set_init_priors(self, initPriorMatrix):
        self.initPriors = initPriorMatrix

    def set_intensity_priors(self, fps, intMeanPriorMatrix, intSigmaPriorMatrix):
        self.fps = fps
        self.nFP = len(fps)
        self.intMeanPriors = intMeanPriorMatrix
        self.intSigmaPriors = intSigmaPriorMatrix
    
    def sample_dyn_pars(self, n):
        ret = zeros([n, self.nDynPars])
        for j in range(self.nDynPars):
            ret[:, j] = uniform(self.dynPriors[j, 0], self.dynPriors[j, 1], n)
        return ret

    def sample_inits(self, n):
        ret = zeros([n, self.nSpecies])
        for j in range(self.nSpecies):
            ret[:, j] = uniform(self.initPriors[j, 0], self.initPriors[j, 1], n)
        return ret

    def sample_int_pars(self, n):
        retMu = zeros([n, self.nFP])
        retSg = zeros([n, self.nFP])
        for j in range(self.nFP):
            retMu[:, j] = uniform(self.intMeanPriors[j, 0], self.intMeanPriors[j, 1], n)
            retSg[:, j] = uniform(self.intSigmaPriors[j, 0], self.intSigmaPriors[j, 1], n)
        return [retMu, retSg]

    # Sampling and perturbing functions
    def sample_perturb_pars(self, n, prior, prev, ids, scales):

        npar = shape(prev)[1]
        ret = zeros((n, npar))

        for i in range(n):
            for j in range(npar):
                ret[i, j] = self.kernel_pert(prev[ids[i], j], prior[j, :], scales[j])
        return ret

    def compute_weights(self, n, weightsPrev, currPar, prevPar, scales):
        ret = zeros([n])
        npar = shape(currPar)[1]
        
        for i in range(n):
            for j in range(n):
                #print "\t\t:", i, j, weightsPrev[i], self.kernel_prob( npar, currPar[i,:], prevPar[j,:], scales )
                ret[i] += weightsPrev[i]*self.kernel_prob(npar, currPar[i, :], prevPar[j, :], scales)

        # Normalise
        ret = ret/float(sum(ret))
        return ret
    
    def kernel_prob(self, npar, currPar, prevPar, scales):
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
            #print "\tx:", x, prior[0], prior[1]
            if x >= float(prior[0]) and x <= float(prior[1]):
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
        ret = zeros([n, 1])

        # create a dictionary of the same form as the data
        for j in range(n):
            sim = {}
            for nt in range(self.ntimePoints):      ##[#threads][#beta][#timepoints][#speciesNumber]
                sim[self.timePoints[nt]] = sims[j, :, nt, :]

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
                    ##y = sorted(self.data[tp][:, 0])
                    ##x = sorted(sim[tp][:, 0])

                    # KS dist
                    #Computes the Kolmogorov-Smirnov statistic on 2 samples.
                    ##rr = ks_2samp(x, y)
                    #Returns [KS-statistic, p-value]
                    ##dist += rr[0]
                    #dist += ks_2samp(self.data[tp][:, 0], sim[tp][:, 0])
                    dist += get_kd_distance1D(self.data[tp], sim[tp], ngrid=100)
                if mode == 2:
                    # Multivariate data, use distance between kernel density estimates
                    dist += get_kd_distance2D(self.data[tp], sim[tp], ngrid=10j)

            ret[j] = dist/len(self.timePoints)

        return ret 
            
    def check_distance(self, n, dists, eps ):
        print "\tcheck_distance : summary : ", percentile(dists, 5), median(dists), percentile(dists, 95)

        ret = zeros([n])
        for j in range(n):
            if dists[j] < eps:
                ret[j] = 1
            else:
                ret[j] = 0
        return ret


    def set_epsilon(self, acceptedDists, alpha):

        accDist = sorted(acceptedDists)
        #print accDist
        #print len(accDist), alpha
        epsilon = accDist[alpha]
        return epsilon

    
    def do_abc_rej(self, model, nbeta, nparticles, eps):
        
        nbatch = 100
        model.create_model_instance(nbeta, self.timePoints)
        self.nSpecies, self.nDynPars = model.get_model_info()
        #print "do_abc_rej : Species/Dynamical parameters in this model:", self.nSpecies, self.nDynPars

        done = False
        ntotsim = 0
        naccepted = 0
        acceptedDynParams = zeros([nparticles,self.nDynPars])
        acceptedInits = zeros([nparticles,self.nSpecies])
        acceptedIntMus = zeros([nparticles,self.nFP])
        acceptedIntSgs = zeros([nparticles,self.nFP])
        acceptedDists = zeros([nparticles])
        while done == False:
            print "\tRunning batch, nsims/nacc:", ntotsim, naccepted

            #Parameters
            dynParameters = self.sample_dyn_pars(nbatch)
            #Initial conditions
            inits = self.sample_inits(nbatch)
            #Intensity parameters
            intMus, intSgs = self.sample_int_pars(nbatch) 
            print "\tDone sampling"
            sims = model.simulate(nbatch, dynParameters, inits, self.fps, intMus, intSgs)
            print "\tDone simulation"
            dists = self.compare_to_data(nbatch, nbeta, sims, self.nFP )
            print 'dists: ', dists
            print "\tDone distance calculation"
            accMask = self.check_distance(nbatch, dists, eps)

            ntotsim += nbatch
            
            for i in range(nbatch):
                if accMask[i] == 1 and naccepted < nparticles:
                    acceptedDynParams[naccepted, :] = dynParameters[i, :]
                    acceptedInits[naccepted, :] = inits[i, :]
                    acceptedIntMus[naccepted, :] = intMus[i, ]
                    acceptedIntSgs[naccepted, :] = intSgs[i, ]
                    acceptedDists[naccepted] = dists[i]
                    naccepted += 1
                if naccepted == nparticles:
                    done = True

        print "do_abc_rej : Completed"
        print "           : Final acceptance rate = ", naccepted/float(ntotsim)
        return [acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs, acceptedDists]

    def do_abc_smc(self, model, nbeta, nparticles, alpha, epsilon_final,
                   outHan, nfp, fps, results_path):

        pop = 0
        finishTotal = False
        #npop = len(epsilons)
        nbatch = 100
        model.create_model_instance(nbeta, self.timePoints)
        self.nSpecies, self.nDynPars = model.get_model_info()
        # do the first population sampling from the prior
        print "do_abc_smc : Population 0"

        acceptedDynParams = self.sample_dyn_pars(nbatch)
        # Initial conditions
        acceptedInits = self.sample_inits(nbatch)
        # Intensity parameters
        acceptedIntMus, acceptedIntSgs = self.sample_int_pars(nbatch)
        print "\tDone sampling"
        sims = model.simulate(nbatch, acceptedDynParams, acceptedInits, self.fps, acceptedIntMus, acceptedIntSgs)
        print "\tDone simulation"
        currDists = self.compare_to_data(nbatch, nbeta, sims, self.nFP)
        #acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs, acceptedDists = self.do_abc_rej(model, nbeta, nparticles, epsilons[0])
        acceptedWeights = ones([nparticles])/float(nparticles)

        while finishTotal == False:
        #for pop in range(1, npop):
            pop += 1
            epsilon = self.set_epsilon(currDists, alpha)
            print "Epsilon current:", epsilon

            currDynParams = zeros([nparticles, self.nDynPars])
            currInits = zeros([nparticles, self.nSpecies])
            currIntMus = zeros([nparticles, self.nFP])
            currIntSgs = zeros([nparticles, self.nFP])
            currDists = zeros(nparticles)

            # calculate the scales for this population
            #(max-min)/2
            scales_dyn = self.calculate_scales(nparticles, acceptedDynParams)
            scales_inits = self.calculate_scales(nparticles, acceptedInits)
            scales_Mus = self.calculate_scales(nparticles, acceptedIntMus)
            scales_Sgs = self.calculate_scales(nparticles, acceptedIntSgs)

            # Rejection stage
            doneRej = False
            ntotsim = 0
            naccepted = 0
            while doneRej == False:
                print "\tRunning batch, nsims/nacc:", ntotsim, naccepted
                #Generates a random sample from a given 1-D array
                ids = numpy.random.choice(nparticles, size=nbatch, replace=True, p=acceptedWeights)
                print "sampled ids:", ids
                dynParameters = self.sample_perturb_pars(nbatch, self.dynPriors, acceptedDynParams, ids, scales=scales_dyn )
                inits         = self.sample_perturb_pars(nbatch, self.initPriors, acceptedInits, ids, scales=scales_inits)
                intMus        = self.sample_perturb_pars(nbatch, self.intMeanPriors, acceptedIntMus, ids, scales=scales_Mus)
                intSgs        = self.sample_perturb_pars(nbatch, self.intSigmaPriors, acceptedIntSgs, ids, scales=scales_Sgs)
                print "\tDone sampling"

                sims = model.simulate(nbatch, dynParameters, inits, self.fps, intMus, intSgs)
                dists = self.compare_to_data(nbatch, nbeta, sims, self.nFP)
                accMask = self.check_distance(nbatch, dists, epsilon)
                ntotsim += nbatch

                for i in range(nbatch):
                    if accMask[i] == 1 and naccepted < nparticles:
                        currDynParams[naccepted, :] = dynParameters[i, :]
                        currInits[naccepted, :] = inits[i, :]
                        currIntMus[naccepted, :] = intMus[i, ]
                        currIntSgs[naccepted, :] = intSgs[i, ]
                        currDists[naccepted] = dists[i]
                        naccepted += 1
                    if naccepted == nparticles:
                        doneRej = True

            print "do_abc_smc : Population", pop, "\tacceptance rate = ", naccepted/float(ntotsim)
            #pop_fold_res_path = 'Population_' + str(pop)
            #os.makedirs('Population_' + str(pop))
            print "Accepted parameters : ", currDynParams

            # update weights
            #column_stack = Stack 1-D arrays as columns into a 2-D array.
            currPar = column_stack((currDynParams, currInits, currIntMus, currIntSgs))
            prevPar = column_stack((acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs))
            all_scales = concatenate((scales_dyn, scales_inits, scales_Mus, scales_Sgs))
            # print shape(currDynParams), shape(currInits), shape(currIntMus), shape(currIntSgs)
            # print shape(currPar), shape(prevPar)
            acceptedWeights = self.compute_weights(nparticles, acceptedWeights, currPar, prevPar, scales=all_scales)
            print "acceptedWeights:"
            print acceptedWeights
            
            # update best estimates
            acceptedDynParams = deepcopy(currDynParams)
            acceptedInits = deepcopy(currInits)
            acceptedIntMus = deepcopy(currIntMus)
            acceptedIntSgs = deepcopy(currIntSgs)

            # write progress out
            outfolder = results_path + "/pop"+repr(pop)
            os.makedirs(outfolder)
            self.write_outputs(outHan, nfp, fps, outfolder, model, nbeta, acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs, acceptedWeights )

            if epsilon <= epsilon_final:
                finishTotal = True
                print "do_abc_smc : Completed successfully"

        return [acceptedDynParams, acceptedInits, acceptedIntMus, acceptedIntSgs, acceptedWeights]

    def write_outputs(self, outHan, nfp, fps, results_path, model_n, nbeta,  accPars, accInit, accMus, accSgs, accWeights ):

        # calculate posterior medians
        medPars = zeros([1, model_n.nparams])
        medInit = zeros([1, model_n.nspecies])
        medMu = zeros([1, nfp])
        medSg = zeros([1, nfp])

        medPars[0, :] = [median(accPars[:, i]) for i in range(model_n.nparams)]
        medInit[0, :] = [median(accInit[:, i]) for i in range(model_n.nspecies)]
        medMu[0, :] = [median(accMus[:, i]) for i in range(nfp)]
        medSg[0, :] = [median(accSgs[:, i]) for i in range(nfp)]
        print "posterior median values dynpar/inits :\n", medPars, "\n", medInit, "\n", medMu, "\n", medSg

        outHan.make_post_hists(results_path, "plot-posteriors-dyn.pdf", accPars, model_n.nparams)
        outHan.make_post_hists(results_path, "plot-posteriors-init.pdf", accInit, model_n.nspecies)
        outHan.make_post_hists(results_path, "plot-posteriors-mu.pdf", accMus, nfp)
        outHan.make_post_hists(results_path, "plot-posteriors-sg.pdf", accSgs, nfp)

        outHan.write_post_params_to_file(results_path, "data-posteriors-dyn.txt", accPars, model_n.nparams)
        outHan.write_post_params_to_file(results_path, "data-posteriors-init.txt", accInit, model_n.nspecies)
        outHan.write_post_params_to_file(results_path, "data-posteriors-mu.txt", accMus, nfp)
        outHan.write_post_params_to_file(results_path, "data-posteriors-sg.txt", accSgs, nfp)

        #Make new model instance
        model_n.create_model_instance(nbeta, self.timePoints)
        res = model_n.simulate(1, medPars, medInit, fps, medMu, medSg)
        #print res

        # convert the output of cuda-sim into a data dictionary
        resDict = model.create_dict(res, self.timePoints)[0]
        # make some plots
        if nfp == 1:
            outHan.make_comp_plot_1D(results_path, "plot-final-fit.pdf", self.data, resDict, self.timePoints)
            outHan.write_post_data_to_file(results_path, "post-final-fit-data.txt", resDict, self.timePoints)
        elif nfp == 2:
            outHan.plot_data_dict_2D(results_path, "plot-final-fit.pdf", resDict, self.timePoints)
            outHan.write_post_data_to_file(results_path, "post-final-fit-data.txt", resDict, self.timePoints)


def read_input(filename):

    document = ElementTree.parse(filename)

    data_f = document.find('data_file')
    data_file = data_f.text
    plot_data_f = document.find('plot_data_file')
    plot_data_file = plot_data_f.text
    model_f = document.find('model_file')
    model_file = model_f.text

    dynPriors = []
    for item in document.find('dynPriors').getchildren():
        dynPriors.append([float(item.find('start').text), float(item.find('end').text)])
    dynPriors = array(dynPriors)

    nparam = len(dynPriors)
    iniPriors = []
    for item in document.find('iniPriors').getchildren():
        iniPriors.append([float(item.find('start').text), float(item.find('end').text)])
    iniPriors = array(iniPriors)

    nspec = len(iniPriors)
    fps = []
    for it in document.find('fps').getchildren():
        fps.append(int(it.find('position').text))
    nfp = len(fps)

    intensMeanPrior = []
    for item in document.find('intensMeanPrior').getchildren():
        intensMeanPrior.append([item.find('start').text, item.find('end').text])
    intensMeanPrior = array(intensMeanPrior)

    intensSigmaPrior = []
    for item in document.find('intensSigmaPrior').getchildren():
        intensSigmaPrior.append([item.find('start').text, item.find('end').text])
    intensSigmaPrior = array(intensSigmaPrior)

    #epsilons = []
    #for item in document.find('epsilons').getchildren():
    #    epsilons.append(float(item.find('epsilon').text))

    eps_f = document.find('epsilon_f')
    epsilon_final = float(eps_f.text)
    alp = document.find('alpha')
    alpha = float(alp.text)

    npart = document.find('npartices')
    nparticles = int(npart.text)
    nb = document.find('nbeta')
    nbeta = int(nb.text)
    alg = document.find('algorithm')
    algorithm = alg.text

    return data_file, plot_data_file, model_file, dynPriors, iniPriors, nparam, nspec, fps, nfp, intensMeanPrior, \
           intensSigmaPrior, epsilon_final, alpha, nparticles, nbeta, algorithm


def main():
    opts, args = getopt.getopt(sys.argv[1:], "hi:o::", ["ifile=", "ofile="])
    for opt, arg in opts:

        if opt in ("-i", "--ifile"):
            print 'Reading input file'
            data_file, plot_data_file, model_file, dynPriorMatrix, initPriorMatrix, nparam, nspec, fps, nfp,\
                intMeanPriorMatrix, intSigmaPriorMatrix, epsilon_final, alp, nparticles, nbeta, algorithm = read_input(arg)

        if opt in ("-o", "--ofile"):
            try:
                os.makedirs(arg)
                results_path = arg
            except:
                print 'Results folder already exists'
    alpha = int(math.ceil(alp * nparticles))-1
    print 'alpha: ', alpha
    print 'epsilon_final: ', epsilon_final
    abcAlg = abc_flow()
    abcAlg.read_data(data_file)

    # plot the data
    outHan = output_handler()
    if nfp == 1:
        outHan.plot_data_dict_1D(results_path, plot_data_file, abcAlg.data, abcAlg.timePoints)
    elif nfp == 2:
        outHan.plot_data_dict_2D(results_path, plot_data_file, abcAlg.data, abcAlg.timePoints)

    # define the model
    model_n = model.model(model_file, nspecies=nspec, nparams=nparam)

    # Set the internal variables
    abcAlg.set_dynamical_priors( dynPriorMatrix)
    abcAlg.set_init_priors(initPriorMatrix)
    abcAlg.set_intensity_priors(fps, intMeanPriorMatrix, intSigmaPriorMatrix)

    if algorithm == 'abc_smc':
        print 'Do abc_smc'
        accPars, accInit, accMus, accSgs, accWeights = abcAlg.do_abc_smc(model_n, nbeta, nparticles, alpha, epsilon_final,
                                                                         outHan, nfp, fps, results_path) # for printing putput within algorithm
    elif algorithm == 'abc_rej':
        accPars, accInit, accMus, accSgs = abcAlg.do_abc_rej(model_n, nbeta, nparticles, epsilon_final)

    # finished

   ##  # calculate posterior medians
##     medPars = zeros([1, model_n.nparams])
##     medInit = zeros([1, model_n.nspecies])
##     medMu = zeros([1, nfp])
##     medSg = zeros([1, nfp])

##     medPars[0, :] = [median(accPars[:, i]) for i in range(model_n.nparams)]
##     medInit[0, :] = [median(accInit[:, i]) for i in range(model_n.nspecies)]
##     medMu[0, :] = [median(accMus[:, i]) for i in range(nfp)]
##     medSg[0, :] = [median(accSgs[:, i]) for i in range(nfp)]
##     print "posterior median values dynpar/inits :", medPars, medInit, medMu, medSg

##     outHan.make_post_hists(results_path, "plot-gardner-2D-posteriors-dyn.pdf", accPars, nparam)
##     outHan.make_post_hists(results_path, "plot-gardner-2D-posteriors-init.pdf", accInit, nspec)
##     outHan.make_post_hists(results_path, "plot-gardner-2D-posteriors-mu.pdf", accMus, nfp)
##     outHan.make_post_hists(results_path, "plot-gardner-2D-posteriors-sg.pdf", accSgs, nfp)

##     outHan.write_post_params_to_file(results_path, "data-posteriors-dyn.txt", accPars, nparam)
##     outHan.write_post_params_to_file(results_path, "data-posteriors-init.txt", accInit, nspec)
##     outHan.write_post_params_to_file(results_path, "data-posteriors-mu.txt", accMus, nfp)
##     outHan.write_post_params_to_file(results_path, "data-posteriors-sg.txt", accSgs, nfp)

##     #Make new model instance
##     model_n.create_model_instance(nbeta, abcAlg.timePoints)
##     res = model_n.simulate(1, medPars, medInit, fps, medMu, medSg)

##     # convert the output of cuda-sim into a data dictionary
##     resDict = model.create_dict(res, abcAlg.timePoints)[0]
##     # make some plots
##     if nfp == 1:
##         outHan.make_comp_plot_1D(results_path, "plot-gene-exp-final-fit.pdf", abcAlg.data, resDict, abcAlg.timePoints)
##         #outHan.make_qq_plots(results_path, "plot-gene-exp-final-fit-qqplots.pdf", abcAlg.data, resDict, abcAlg.timePoints)
##         outHan.write_post_data_to_file(results_path, "post_final_fit_data.txt", resDict, abcAlg.timePoints)
##     elif nfp == 2:
##         outHan.plot_data_dict_2D(results_path, "plot-gard-final-fit.pdf", resDict, abcAlg.timePoints)
##         #outHan.make_qq_plots(results_path, "plot-gene-exp-final-fit-qqplots.pdf", abcAlg.data, resDict, abcAlg.timePoints)
##         outHan.write_post_data_to_file(results_path, "post_final_fit_data.txt", resDict, abcAlg.timePoints)

main()
