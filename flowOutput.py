from numpy import *
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rc('xtick', labelsize=4) 
matplotlib.rc('ytick', labelsize=4) 

class output_handler:
    def __init__(self):
        pass

    def plot_sim_res_1D(self, file, results, ntimes): 
        pp = PdfPages(file)
        nBins = 50
        for jt in range(ntimes):
            plt.subplot(4,5,jt+1)
            plt.hist( results[0,:,jt,0], nBins )
            plt.xlim(0,1500)
            # plt.xscale('log')
        pp.savefig()
        plt.close()
        pp.close()

    def plot_data_dict_1D(self, results_path, file_n, data, timepoints):
        pp = PdfPages(results_path+'/'+file_n)
        nBins = 50

        cc = 0
        for tp in timepoints:
            plt.subplot(4,5,cc+1)
            plt.hist( data[tp], nBins )
            plt.xlim(0,1500)
            cc += 1
        pp.savefig()
        plt.close()
        pp.close()

    def plot_data_dict_2D(self, results_path, file_n, data, timepoints):

        pp = PdfPages(results_path+'/'+file_n)
        cc = 0
        for tp in timepoints:
            xmin, xmax = -3, 3
            ymin, ymax = -3, 3
            xx, yy = mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = vstack([xx.ravel(), yy.ravel()])
            values = vstack([ log10(data[tp][:,0]), log10(data[tp][:,1]) ])
            kernel = st.gaussian_kde(values)
            f = reshape(kernel(positions).T, xx.shape)

            ax = plt.subplot(4,5,cc+1)
            ax.contourf(xx, yy, f, cmap='Blues')
            ax.set_xlim([-1,3])
            ax.set_ylim([-1,3])
            cc += 1
        pp.savefig()
        plt.close()
        pp.close()

    def plot_data_comb_2D(self, results_path, file_n, data, fit, timepoints):

        pp = PdfPages(results_path+'/'+file_n)
        cc = 0
        for tp in timepoints:
            xmin, xmax = -3, 3
            ymin, ymax = -3, 3

            xx, yy = mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = vstack([xx.ravel(), yy.ravel()])
            values = vstack([ log10(data[tp][:, 0]), log10(data[tp][:, 1])])
            kernel = st.gaussian_kde(values)
            f = reshape(kernel(positions).T, xx.shape)

            xxf, yyf = mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions_f = vstack([xxf.ravel(), yyf.ravel()])
            values_f = vstack([log10(fit[tp][:, 0]), log10(fit[tp][:, 1])])
            kernel_f = st.gaussian_kde(values_f)
            ff = reshape(kernel_f(positions_f).T, xxf.shape)

            ax = plt.subplot(4, 5, cc+1)
            ax.contourf(xx, yy, f, cmap='Blues')
            ax.contourf(xxf, yyf, ff, cmap='Reds')

            ax.set_xlim([-1, 3])
            ax.set_ylim([-1, 3])
            cc += 1
        pp.savefig()
        plt.close()
        pp.close()

    def make_qq_plots(self, results_path, file_n, data, sims, timepoints, ind=0):
        pp = PdfPages(results_path+'/'+file_n)
        cc = 0
        for tp in timepoints:
            plt.subplot(4, 5, cc+1)
            y = sorted(data[tp][:, ind])
            x = sorted(sims[tp][:, ind])

            plt.plot( x, y, ".")
            ax = plt.gca()
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
          
            lims = [ min([xlim,ylim]), max([xlim,ylim]) ]
            plt.plot(lims, lims, 'r-', linewidth=3)
            cc += 1
        pp.savefig()
        plt.close()
        pp.close()

    def make_comp_plot_1D(self, results_path, file_n, data, sims, timepoints, ind=0):
        pp = PdfPages(results_path+'/'+file_n)
        cc = 0
        nBins = 20
        for tp in timepoints:

            plt.subplot(4, 5, cc+1)
            y = data[tp][:, ind]
            x = sims[tp][:, ind]

            plt.hist(x, nBins, alpha=0.25, label="data")
            plt.hist(y, nBins, alpha=0.25, label="fit")
            #plt.legend(loc='upper right')
            cc += 1

        pp.savefig()
        plt.close()
        pp.close()


    def make_post_hists(self, results_path, file_n, post, npar):
        pp = PdfPages(results_path+'/'+file_n)
        #npar = len(pars)

        for i in range(npar):
            plt.subplot(1, npar, i)
            plt.hist(post[:, i-1], normed=1)
        pp.savefig()
        plt.close()
        pp.close()

    def write_post_params_to_file(self, results_path, file_n, post, pars):
        savetxt(results_path+'/'+file_n, post, delimiter=' ')


    def write_post_data_to_file(self, results_path, file_n, data, timepoints):
        values = []
        for tp in timepoints:
            for sp in range(len(data[tp][:, 0])):
                values.append([tp, data[tp][sp, 0], data[tp][sp, 1]])
        savetxt(results_path+'/'+file_n, asarray(values), delimiter=' ')
