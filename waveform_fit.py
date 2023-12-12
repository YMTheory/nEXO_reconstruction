# The main problem on the waveform deconvolution is determining how many clusters contributing to this event.
# This can be applied on one single strip while also a combined analysis with all strips in one event.

# author: miaoyu@slac.stanford.edu
# date: Nov-Dec, 2023

import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import re

from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.util import describe

from SignalCalculator import SignalCalculator
from toy_digitization import digitization
from toyMC_loader import toyMC_loader
from generate_PDFs import pdf_generator

class fitter():
    def __init__(self, SamplingFrequency) -> None:
        self.SamplingFrequency = SamplingFrequency      # unit: MHz
        self.SamplingInterval  = 1 / SamplingFrequency  # unit: us

        self.digi = digitization(SamplingFrequency=SamplingFrequency)

        self.gridPDFs_induction = {}
        self.gridPDF_induction_filename = './gridPDFs_induction_nonuniform_simplified_new2.h5'
        #self.gridPDF_induction_filename = './gridPDFs_induction_nonuniform.h5'
        #self.gridPDF_induction_filename = './gridPDFs_induction_test.h5'
        self.gridPDFs_collection = {}
        self.gridPDF_collection_filename = './gridPDFs_collection_test.h5'
        self.gridPDFs_time = None
        self.pdf_length = 0
        self.x_step = 6.
        self.x_min = -48
        self.x_max = 48
        self.y_step = 6.
        self.y_min = -30
        self.y_max = 30
        
        self.v_drift = 1.70 # um / ms
        
        # As the uniform gridPDFs performs badly, try non-uniform dividing.
        self.xbonds = np.array([-48, -44, -40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10.0, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48], dtype=float)
        self.ybonds = np.array([-30, -28, -26, -24, -22, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10.0, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30], dtype=float)

        self.toyMC_loader = toyMC_loader()
        self.gen = pdf_generator()


    def dummy_fixed_model(self, t, p0):
        t0, x0, y0, z0, q0, t1, x1, y1, z1, q1 = p0
        preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q0, x0, y0, z0)
        self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
        cryo_time0, cryo_amp0 = self.digi.cryoTime, self.digi.cryoAmp
        preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q1, x1, y1, z1)
        self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
        cryo_time1, cryo_amp1 = self.digi.cryoTime, self.digi.cryoAmp
        
        # consider time shift:
        tmax = 2000 # maximum estimation of drift time as 2 ms.
        sumWFLen = int(tmax / self.digi.SamplingInterval)

        cryoAmp0_shifted = [0 for _ in range(int(t0/self.SamplingInterval))] + cryo_amp0 + [0 for _ in range(sumWFLen - len(cryo_amp0) - int(t0/self.SamplingInterval))]
        cryoAmp1_shifted = [0 for _ in range(int(t1/self.SamplingInterval))] + cryo_amp1 + [0 for _ in range(sumWFLen - len(cryo_amp1) - int(t1/self.SamplingInterval))]

        cryoAmp_summed = [0 for _ in range(sumWFLen)]

        for idx in range(sumWFLen):
            cryoAmp_summed[idx] += cryoAmp0_shifted[idx]
            cryoAmp_summed[idx] += cryoAmp1_shifted[idx]

        time = np.arange(sumWFLen) * 0.5
        return np.interp(t, time, cryoAmp_summed)



    def dummy_fixed_fitting(self, x, y, p0):
        t0, x0, y0, z0, q0, t1, x1, y1, z1, q1 = p0
        q0, q1 = 1, 1
        preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q0, x0, y0, z0)
        self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
        cryo_time0, cryo_amp0 = self.digi.cryoTime, self.digi.cryoAmp
        preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q1, x1, y1, z1)
        self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
        cryo_time1, cryo_amp1 = self.digi.cryoTime, self.digi.cryoAmp

        # consider time shift:
        tmax = 2000 # maximum estimation of drift time as 2 ms.
        sumWFLen = int(tmax / self.digi.SamplingInterval)

        cryoAmp0_shifted = [0 for _ in range(int(t0/self.SamplingInterval))] + cryo_amp0 + [0 for _ in range(sumWFLen - len(cryo_amp0) - int(t0/self.SamplingInterval))]
        cryoAmp1_shifted = [0 for _ in range(int(t1/self.SamplingInterval))] + cryo_amp1 + [0 for _ in range(sumWFLen - len(cryo_amp1) - int(t1/self.SamplingInterval))]
        #cryoAmp2_shited = [0 for _ in range(int(t2/self.SamplingInterval))] + cryo_amp2 + [0 for _ in range(sumWFLen - len(cryo_amp2) - int(t2/self.SamplingInterval))]
        
        def model(t, q0, q1):
            tmax = 2000 # maximum estimation of drift time as 2 ms.
            sumWFLen = int(tmax / self.digi.SamplingInterval)
            
            cryoAmp_summed = [0 for _ in range(sumWFLen)]

            for idx in range(sumWFLen):
                cryoAmp_summed[idx] += q0 * cryoAmp0_shifted[idx]
                cryoAmp_summed[idx] += q1 * cryoAmp1_shifted[idx]

            time = np.arange(sumWFLen) * 0.5
            return np.interp(t, time, cryoAmp_summed)

        yerr = np.ones(len(y))
        t_start = time.time()
        c = LeastSquares(x, y, yerr, model, loss='soft_l1')
        m = Minuit(c, q0=p0[4], q1=p0[9])
        m.migrad()
        t_end = time.time()
        print(f'Total fitting time consumed: {t_end - t_start}.')
        return m

    def dummy_fitting(self, x, y, p0):
        '''
        Suppose we know the exact number of clusters contributing to these waveforms. Only considering effects from noises and diffusion?
        '''

        def model(t, t0, x0, y0, z0, q0, t1, x1, y1, z1, q1):
            preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q0, x0, y0, z0)
            self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
            cryo_time0, cryo_amp0 = self.digi.cryoTime, self.digi.cryoAmp
            preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q1, x1, y1, z1)
            self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
            cryo_time1, cryo_amp1 = self.digi.cryoTime, self.digi.cryoAmp
            #preSampleTime, preSampleAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q2, x2, y2, z2)
            #self.digi.convolve_asic_response(preSampleTime, preSampleAmp)
            #cryo_time2, cryo_amp2 = self.digi.cryoTime, self.digi.cryoAmp

            # consider time shift:
            tmax = 2000 # maximum estimation of drift time as 2 ms.
            sumWFLen = int(tmax / self.digi.SamplingInterval)

            cryoAmp_summed = [0 for _ in range(sumWFLen)]
            cryoAmp0_shifted = [0 for _ in range(int(t0/self.SamplingInterval))] + cryo_amp0 + [0 for _ in range(sumWFLen - len(cryo_amp0) - int(t0/self.SamplingInterval))]
            cryoAmp1_shifted = [0 for _ in range(int(t1/self.SamplingInterval))] + cryo_amp1 + [0 for _ in range(sumWFLen - len(cryo_amp1) - int(t1/self.SamplingInterval))]
            #cryoAmp2_shited = [0 for _ in range(int(t2/self.SamplingInterval))] + cryo_amp2 + [0 for _ in range(sumWFLen - len(cryo_amp2) - int(t2/self.SamplingInterval))]
            
            for idx in range(sumWFLen):
                cryoAmp_summed[idx] += cryoAmp0_shifted[idx]
                cryoAmp_summed[idx] += cryoAmp1_shifted[idx]

            time = np.arange(sumWFLen) * 0.5

            return np.interp(t, time, cryoAmp_summed)
        
        yerr = np.ones(len(y))
        t_start = time.time()
        c = LeastSquares(x, y, yerr, model, loss='soft_l1')
        m = Minuit(c, t0=p0[0], x0=p0[1], y0=p0[2], z0=p0[3], q0=p0[4], t1=p0[5], x1=p0[6], y1=p0[7], z1=p0[8], q1=p0[9])
        m.print_level = 2
        m.fixed['x0'] = True
        m.fixed['y0'] = True
        m.fixed['z0'] = True
        m.fixed['t0'] = True
        m.fixed['x1'] = True
        m.fixed['y1'] = True
        m.fixed['z1'] = True
        m.fixed['t1'] = True
        m.migrad()
        t_end = time.time()
        print(f'Total fitting time consumed: {t_end - t_start}.')
        return m.values


    def load_gridPDFs(self):
        self.gridPDFs_collection = {}
        self.gridPDFs_induction = {}
        with h5py.File(self.gridPDF_induction_filename, 'r') as f:
            for dset_name in f.keys():
                dataset = f[dset_name]
                pdf = dataset[:]
                self.gridPDFs_induction[dset_name] = pdf
        with h5py.File(self.gridPDF_collection_filename, 'r') as f:
            for dset_name in f.keys():
                dataset = f[dset_name]
                pdf = dataset[:]
                self.gridPDFs_collection[dset_name] = pdf

        ## All PDFs are extended, 50 more points before and 50 more points after
        for ky, pdf in self.gridPDFs_collection.items():
            pdf = np.append(np.zeros(50), pdf)
            pdf = np.append(pdf, np.zeros(50))
            self.gridPDFs_collection[ky] = pdf
            self.pdf_length = len(pdf)

        for ky, pdf in self.gridPDFs_induction.items():
            pdf = np.append(np.zeros(50), pdf)
            pdf = np.append(pdf, np.zeros(50))
            self.gridPDFs_induction[ky] = pdf

        self.gridPDFs_time = np.arange(self.pdf_length) * self.SamplingInterval


    def PDF_interpolation(self, x, y):
        ## Using the 4-fold symmetry in PDFs, compare the absolute value directly.
        if np.abs(x) < self.x_min or np.abs(x) > self.x_max or np.abs(y) < self.y_min or np.abs(y) > self.y_max:
            # out the ROI
            print('Fitting x or y parameters have run out of the ROI.')
            return np.zeros(self.pdf_length)
        else:
            #print(x, y)
            x_left, x_right, y_down, y_up = None, None, None, None
            ### New method, for non-uniform grids PDF
            for i in range(len(self.xbonds)-1):
                if self.xbonds[i] <= x < self.xbonds[i+1]:
                    x_left, x_right = self.xbonds[i], self.xbonds[i+1]
            for i in range(len(self.ybonds)-1):
                if self.ybonds[i] <= y < self.ybonds[i+1]:
                    y_down, y_up = self.ybonds[i], self.ybonds[i+1]

            if None in (x_left, x_right, y_down, y_up):
                print(f'Error!! {x} within ({x_left}, {x_right}), {y} within ({y_down}, {y_up}).')

            #print(x_left, x_right, y_down, y_up)
            name00 = f'x{x_left}y{y_down}'
            name01 = f'x{x_left}y{y_up}'
            name10 = f'x{x_right}y{y_down}'
            name11 = f'x{x_right}y{y_up}'

            flag00, flag01, flag10, flag11 = False, False, False, False
            if name00 in self.gridPDFs_induction.keys():
                flag00 = True
                f00 = self.gridPDFs_induction[name00]
            if name01 in self.gridPDFs_induction.keys():
                flag01 = True
                f01 = self.gridPDFs_induction[name01]
            if name10 in self.gridPDFs_induction.keys():
                flag10 = True
                f10 = self.gridPDFs_induction[name10]
            if name11 in self.gridPDFs_induction.keys():
                flag11 = True
                f11 = self.gridPDFs_induction[name11]
            
            if flag00 and flag01 and flag10 and flag11:
                wx = (x_right - x)/(x_right - x_left)
                wy = (y_up - y)/(y_up - y_down)
                #print(x, x_left, x_right, wx)
                f = f00 * wx * wy + f01 * wx * (1-wy) + f10 * (1-wx) * wy + f11 * (1-wx) * (1-wy)  
            #return np.interp(t, self.gridPDFs_time, f)

            else:
                if flag00:
                    f = f00
                else:
                    if flag01:
                        f = f01
                    else:
                        if flag10:
                            f = f10
                        else:
                            if flag11:
                                f = f11
                            else:
                                print(f'Error! There is no boundary points for ({x}, {y})!!! CHECK-REQUIRED!!!')
                                f = self.gridPDFs_induction['x0.0y0.0']
            return f
        
        

    def dummy_oneCollectionWF_fitting(self, x, y, p0):
        def model(t, t0, z0, q0):
            v = 1.70
            dt_drift = z0/v - 100/v
            dt = dt_drift + t0 - 25 # this hard-coded 25 ms corresponds to the waveform pdf extension implemented above.
            f0 = self.gridPDFs_collection['coll'] * q0
            
            return np.interp(t-dt, self.gridPDFs_time, f0)
        
        yerr = np.ones(len(y)) * (100 / int(40000./4096))
        c = LeastSquares(x, y, yerr, model, loss='soft_l1')
        m = Minuit(c, t0=p0[0], z0=p0[1], q0=p0[2])
        m.print_level = 1
        #m.limits = [(-10, 10), (0, 1300), (p0[2]*0.9, p0[2]*1.1)]
        m.limits = [(-500, 500), (p0[1]*0.95, p0[1]*1.05), (p0[2]*0.8, p0[2]*1.2)]
        m.fixed['t0'] = True
        m.migrad()

        return m
            
    
    def dummy_twoCollectionWF_fitting(self, x, y, p0):
        def model(t, z0, q0, z1, q1):
            v = 1.70
            dt_drift = z0/v - 100/v
            dt0 = dt_drift - 25 # this hard-coded 25 ms corresponds to the waveform pdf extension implemented above.
            f0 = self.gridPDFs_collection['coll'] * q0
            
            dt_drift = z1/v - 100/v
            dt1 = dt_drift - 25
            f1 = self.gridPDFs_collection['coll'] * q1
            
            return np.interp(t-dt0, self.gridPDFs_time, f0) + np.interp(t-dt1, self.gridPDFs_time, f1)
        
        yerr = np.ones(len(y)) * (100 / int(40000./4096))
        c = LeastSquares(x, y, yerr, model, loss='soft_l1')
        m = Minuit(c, z0=p0[0], q0=p0[1], z1=p0[2], q1=p0[3])
        m.print_level = 1
        m.migrad()

        return m

    def dummy_oneInductionWF_fitting(self, x, y, p0):
        def model(t, t0, x0, y0, z0, q0):
            v = 1.70
            dt_drift = z0/v - 100/v
            dt = dt_drift + t0 -25
            f0 = self.PDF_interpolation(x0, y0) * q0
            return np.interp(t-dt, self.gridPDFs_time, f0)
        
        yerr = np.ones(len(y)) * (100 / int(40000./4096))

        c = LeastSquares(x, y, yerr, model, loss='soft_l1')
        m = Minuit(c, t0=p0[0], x0=p0[1], y0=p0[2], z0=p0[3], q0=p0[4])
        m.print_level = 1
        m.limits = [(-500, 500), (p0[1]*0.7, p0[1]*1.3), (3, p0[2]*1.3), (p0[3]*0.95, p0[3]*1.05), (p0[4]*0.9, p0[4]*1.1)]
        m.fixed['t0'] = True
        try:
            m.migrad()
            return m
        except:
            print('One failed fitting...')
            return False
        
    def dummy_oneCollectionWF_fitting_plot(self, x, y, p0):
        def model(t, t0, z0, q0):
            v = 1.70
            dt_drift = z0/v - 100/v
            dt = dt_drift + t0 - 25 # this hard-coded 25 ms corresponds to the waveform pdf extension implemented above.
            f0 = self.gridPDFs_collection['coll'] * q0
            
            return np.interp(t-dt, self.gridPDFs_time, f0)

        _, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'o', ms=5, label='data')
        ax.plot(x, model(x, *p0), '-', lw=2, label='best fit')

        ax.set_xlabel(r'drift time [$\mu$s]', fontsize=12)
        ax.set_ylabel('adc', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(prop={'size':12})
        ax.set_xlim(x[-100], x[-1])
        plt.show()

    def dummy_oneInductionWF_fitting_plot(self, x, y, p0, e0):
        def model(t, t0, x0, y0, z0, q0):
            v = 1.70
            dt_drift = z0/v - 100/v
            dt = dt_drift + t0 -25
            f0 = self.PDF_interpolation(x0, y0) * q0
            return np.interp(t-dt, self.gridPDFs_time, f0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'o', ms=5, label='data')
        ax.plot(x, model(x, *p0), '-', lw=2, label='best fit')

        preT, preA = SignalCalculator.ComputeChargeWaveformOnStripWithIons(1e5, 1.5, 4.0, 220.)
        self.digi.convolve_asic_response(preT, preA)
        self.digi.generate_noise(len(self.digi.cryoAmp))
        self.digi.quantization(self.digi.cryoAmp, self.digi.asic_noise_amp, 40000.)
        ax.plot(np.arange(len(self.digi.cryoAmp))*0.5, self.digi.outputWF, ':', label='truth')
        
        ax.text(100, -250, r'x=1.5 mm, $x_{fit}=$' + f'{p0[1]:.3f}' + r'$\pm$' + f'{e0[1]:.3f}' + ' mm.', fontsize=12)
        ax.text(100, -450, r'y=4.0 mm, $y_{fit}=$' + f'{p0[2]:.3f}' + r'$\pm$' + f'{e0[2]:.3f}' + ' mm.', fontsize=12)
        ax.text(100, -650, r'x=220.0 um, $z_{fit}=$' + f'{p0[3]:.3f}'+ r'$\pm$' + f'{e0[3]:.3f}' + ' um.', fontsize=12)
        print(p0[4], e0[4])
        ax.text(100, -850, r'q=1.0e5, $q_{fit}=$' + f'{p0[4]*1e5:.3e}' + r'$\pm$' + f'{e0[4]*1e5:.3f}', fontsize=12)
        
        ax.set_xlabel(r'drift time [$\mu$s]', fontsize=12)
        ax.set_ylabel('adc', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(prop={'size':12})
        ax.set_xlim(x[-100], x[-1])
        plt.show()

    def pointCharge_multiChannels_fit(self, filename, evtNo, t0, x0, y0, z0, q0, xc, yc):
        self.toyMC_loader.toyMC_filename = filename
        self.toyMC_loader.load_h5file_multiChannels(evtNo)
        waveforms = self.toyMC_loader.waveform_multiChannel_oneEvt
        fit_x, fit_y, fit_z, fit_q = [], [], [], []
        chas = []
        for name, wf in waveforms.items():
            chas.append(name)
            time = np.arange(len(wf)) * 0.5
            if name.startswith('x'):
                # This is a x-axis aligned channel
                cord = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', name)
                x, y = int(cord[0]), int(cord[1])
                if x == 0 and y == 0:
                    # hard coded, collection channel
                    m = self.dummy_oneCollectionWF_fitting(time, wf, [t0, z0, q0])
                    fit_x.append(-1000)
                    fit_y.append(-1000)
                    fit_z.append(m.values['z0'])
                    fit_q.append(m.values['q0'])
                else:
                    m = self.dummy_oneInductionWF_fitting(time, wf, [t0, np.abs(x0-x), np.abs(y0-y), z0, q0])
                    fit_x.append(m.values['x0']+x)
                    fit_y.append(m.values['y0']+y)
                    fit_z.append(m.values['z0'])
                    fit_q.append(m.values['q0'])
                    
            else:
                # This is a y-axis aligned channel, need to rotate
                cord = re.findall(r'\d+\.\d+|\d+', name)
                x, y = int(cord[0]), int(cord[1])
                m = self.dummy_oneInductionWF_fitting(time, wf, [t0, np.abs(y0-y), np.abs(x0-x), z0, q0])
                fit_x.append(m.values['y0']+y)
                fit_y.append(m.values['x0']+x)
                fit_z.append(m.values['z0'])
                fit_q.append(m.values['q0'])

        return chas, fit_x, fit_y, fit_z, fit_q
    
    
    
    
    def oneCluster_fitting(self, time_arr, wf_arr, t0, x0, y0, z0, Q0, sx_arr, sy_arr, ystrip_arr):
        '''
        Fitting works for both induction and collection channels as well.
        time, wf are the time series and adc series to be fitted.
        sx, sy are the coordinate of the center of the strip, also ystrip flag is to identify if it is a y-along strip.
        '''

        def create_models(sx, sy, ystrip):
            def one_model(t, t0, x0, y0, z0, Q0):
                if not ystrip:
                    px, py = x0 - sx, y0 - sy
                else:
                    px, py = y0 - sy, x0 - sx

                px, py = np.abs(px), np.abs(py)
                                
                flag = self.gen._is_point_onStrip((px, py))
                
                if flag:
                    f0 = self.gridPDFs_collection['coll'] * Q0
                else:   
                    f0 = self.PDF_interpolation(px, py) * Q0 

                dt_drift = z0 / self.v_drift - 100. / self.v_drift # 100 from PDF generator.
                dt = dt_drift + t0 - 25
                return np.interp(t-dt, self.gridPDFs_time, f0)
                
            return one_model

            
        models_list = []
        for i in range(len(sx_arr)):
            models_list.append(create_models(sx_arr[i], sy_arr[i], ystrip_arr[i]))
            
        least_squres = []            
        for i in range(len(sx_arr)):
            t, wf = time_arr[i], wf_arr[i]
            model = models_list[i]
            # loop all channels and combine fitting
            wferr = np.ones(len(wf)) * (100 / int(40000./4096.))
            #ls = LeastSquares(t, wf, wferr, model, loss='soft_l1')
            ls = LeastSquares(t, wf, wferr, model)
            least_squres.append(ls)
        least_squres_total = least_squres[0]
        for i in range(len(least_squres)-1):
            least_squres_total += least_squres[i+1]
            
        print(f'{describe(least_squres_total)=}.')
        
        ############ Coordinates quick checking #############
        ii = 0
        for sx, sy, ystrip in zip(sx_arr, sy_arr, ystrip_arr):
            ty = 'x'
            dx, dy = x0 - sx, y0 - sy
            if ystrip:
                ty = 'y'
                dx, dy = y0 - sy, x0 - sx
            print(f'Local coordinates for the charge on strip{ii} ({ty}) is ({dx}, {dy}).')
            ii += 1
            
        m = Minuit(least_squres_total, t0=t0, x0=x0, y0=y0, z0=z0, Q0=Q0)
        m.print_level = 1
        
        #xmin, xmax = x0 * 0.95, x0*1.05
        #ymin, ymax = y0 * 0.5, y0*1.50
        #if x0 == 0:
        #    xmin, xmax = -1, 1
        #if y0 == 0:
        #    ymin, ymax = -1, 1
        
        dx = 15.0 # mm
        dy = 4.0 # mm
        xmin, xmax = x0 - dx, x0 + dx
        ymin, ymax = y0 - dy, y0 + dy
         
            
        #print('x-range: ', xmin, xmax)
        #print('y-range: ', ymin, ymax) 
        m.limits = [(-500, 500), (-42, 42), (ymin, ymax), (0.8*z0, 1.2*z0), (0.5*Q0, 1.5*Q0)]
        m.fixed['t0'] = True
        m.fixed['x0'] = False
        m.fixed['y0'] = False
        m.fixed['z0'] = False
        m.fixed['Q0'] = False
        m.migrad()
        
        return m
        
        
        
            