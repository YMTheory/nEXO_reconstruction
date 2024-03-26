import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import sys
sys.path.append("/fs/ddn/sdf/group/nexo/users/miaoyu/Reconstruction/Softwares/nEXO_reconstruction/")

from scripts.stencilPCD_fitting import pcd_fitter
from scripts.realistic_PDFs_generator import generator


class drawer():

    def __init__(self):
        self.fitter = pcd_fitter()
        self.gen    = generator(z0=-622.0, xy_step=0.5, z_step=0.5, charge_cubic_L=20., charge_cubic_H=20.)

        self.fit_t  = 0.0
        self.fit_x  = 0.0
        self.fit_y  = 0.0
        self.fit_Q  = 0.0
        self.true_Q = 0.0

    def draw_fitVals_1Dim(self, fitVals, labels, xlabel):
        fig, ax = plt.subplots(figsize=(7, 5))
        
        low_range, high_range = np.min([np.min(arr) for arr in fitVals]), np.max([np.max(arr) for arr in fitVals])
    
        for fitval, lb in zip(fitVals, labels):
            ax.hist(fitval, bins=50, range=(low_range, high_range), histtype='step', label=lb)
        ax.legend()
        ax.set_xlabel(xlabel)
        plt.tight_layout()
        plt.show()
        return fig, ax
    

    def draw_raw_charge_distribution(self, fits, itgs, truths, relQs, itgrelQs, labels):
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        fits_min, fits_max = np.min([np.min(arr) for arr in fits]), np.max([np.max(arr) for arr in fits])
        itgs_min, itgs_max = np.min([np.min(arr) for arr in itgs]), np.max([np.max(arr) for arr in itgs])
        truths_min, truths_max = np.min([np.min(arr) for arr in truths]), np.max([np.max(arr) for arr in truths])
        max_bin = np.max([fits_max, truths_max, itgs_max])
        min_bin = np.min([fits_min, truths_min, itgs_min])

        min_fitrel_bin, max_fitrel_bin = np.min([np.min(arr) for arr in relQs]), np.max([np.max(arr) for arr in relQs])
        min_itgrel_bin, max_itgrel_bin = np.min([np.min(arr) for arr in itgrelQs]), np.max([np.max(arr) for arr in itgrelQs])
        
    
        for fit, itg, tru, rel, itgrel, lb in zip(fits, itgs, truths, relQs, itgrelQs, labels):
            ax[0, 0].hist(fit,      bins=50, range=(min_bin, max_bin),               linestyle='-', histtype='step', label='fit, '+lb       )
            ax[0, 0].hist(tru,      bins=50, range=(min_bin, max_bin),               linestyle=':', histtype='step', label='truth, '+lb     )
            ax[0, 1].hist(itg,      bins=50, range=(min_bin, max_bin),               linestyle='-', histtype='step', label='integral, '+lb  )
            ax[0, 1].hist(tru,      bins=50, range=(min_bin, max_bin),               linestyle=':', histtype='step', label='truth, '+lb     )
            ax[1, 0].hist(rel,      bins=50, range=(min_fitrel_bin, max_fitrel_bin), linestyle='-', histtype='step', label='fit, '+lb       )
            ax[1, 1].hist(itgrel,   bins=50, range=(min_itgrel_bin, max_itgrel_bin), linestyle='-', histtype='step', label='integral, '+lb  )
    
        for i in range(2):
            ax[0, i].set_xlabel('charge')
            ax[0, i].legend()
            ax[1, i].set_xlabel('(rec - true) / true [%]')
            ax[1, i].legend()

        plt.tight_layout()
        plt.show()



    
    def draw_fitVals_2Dim(self, fitVals1, fitVals2, labels, xlabel, ylabel):
        fig, ax = plt.subplots( figsize=(7, 5) )
        for v1, v2, lb in zip(fitVals1, fitVals2, labels):
            ax.scatter(v1, v2, s=10, alpha=0.3, label=lb)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.show()
        return fig, ax
    

    def config_fitter(self, filename, evtid, inductive, noise, fine, sscut, threshold):
        self.fitter._set_offline_filename(filename)
        self.fitter._set_event_id(evtid)
        self.fitter._set_loading_nevents(500)
        self.fitter._set_noise_flag(noise)
        self.fitter._set_fit_inductive_flag(inductive)
        self.fitter._set_pdf_fine(fine)
        self.fitter._set_SS_cut(sscut)
        self.fitter._set_amplitude_threshold(threshold)
    
        self.fitter.load_one_event()


    def config_fitVals(self, tf, xf, yf, Qf, Qt):
        self.fit_t = tf
        self.fit_x = xf
        self.fit_y = yf
        self.fit_Q = Qf
        self.true_Q = Qt
    
    
    def draw_channel_waveforms(self):
    
        fig = self.fitter.onePC_nofitting(self.fit_t, self.fit_x, self.fit_y, self.fit_Q/1.e5, draw=True)
        plt.show()
        return fig


    
    def fitting_strip_charge(self):
        self.gen.x0 = self.fit_x
        self.gen.y0 = self.fit_y
        self.gen.q0 = self.fit_Q

        Q_fit = []
        for i in range(self.fitter.fit_nchannels):
            strip_Q, totQ, collect_nPCD = self.gen.collection_charges_onOneChannel(self.fitter.strip_x_array[i], self.fitter.strip_y_array[i], self.fitter.strip_type_array[i])
            Q_fit.append(strip_Q)
        Q_fit = np.array(Q_fit)

        itg_Qrec, Qtrue, itg_Qrec_oneChannel = self.baseline_reconstruction()

        print('============================================')
        print(f'The point charge now is located at ({self.fit_x:.3f}, {self.fit_y:.3f}) mm.')
        print(tabulate([[self.true_Q, itg_Qrec, self.fit_Q]], headers=['true',  'simple integral', 'combined fitter']))
        print('============================================')
        print(tabulate([[self.fitter.strip_x_array[i], self.fitter.strip_y_array[i], self.fitter.strip_x_array[i] - self.fit_x, self.fitter.strip_y_array[i] - self.fit_y, self.fitter.strip_type_array[i], self.fitter.strip_charge_array[i], Q_fit[i], itg_Qrec_oneChannel[i]] for i in range(self.fitter.fit_nchannels)], headers=['x', 'y', 'dx', 'dy', 'IsXstrip', 'true Q', 'fit Q', 'integral Q']))
        print('============================================')



    def baseline_reconstruction(self):
        builder = self.fitter.builder
        event_Qrec, event_Qtrue, = builder.simple_reconstruction()
        channel_rec_q = builder.channel_rec_q_all
        return event_Qrec, event_Qtrue, channel_rec_q
        






