# This script generate toy waveforms by considering ASIC waveform response and noises.
# Author: miaoyu@slac.stanford.edu
# Date: Nov, 2023

import numpy as np
import uproot as up

from SignalCalculator import SignalCalculator

class digitization():
    def __init__(self, SamplingFrequency) -> None:
        self.SamplingFrequency = SamplingFrequency      # unit: MHz
        self.SamplingInterval  = 1./SamplingFrequency   # unit: us

        self.asic_response_flag = False
        self.asic_model_file = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/cryo_asic_model.root'
        self.asic_response_amp = []

        self.asic_noise_file = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/noise_lib_1_2_us_100e.root'
        self.asic_noise_amp  = []
        self.asic_noises = []
        self.asic_noise_psd_file = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/noise_unshape.root'

        self.preTime, self.preAmp = None, None
        self.cryoTime = []
        self.cryoAmp  = []
        self.outputWF = []
        self.fTruth   = []

        self.DT = 53. # cm2/s
        self.DL = 24.8 #cm2/s
        self.v_drift = 1.70 # mm/us

    def load_asic_response(self, bname='1_2_us_28.6'):
        self.asic_response_amp = []
        with up.open(self.asic_model_file) as f:
            tree = f['cryo_asic_model']
            brh  = tree[bname].array()[0]
            for elm in brh:
                self.asic_response_amp.append(elm)
        self.asic_response_flag = True


    def convolve_asic_response(self, preSampleTime, preSampleAmp):
        '''
        In nEXO offline, the pre-sampling points are not supposed uniformly distributed on the time axis.
        The preSampleWF is a charge waveform.
        One should firstly over-sampling the waveform to 50 MHz sampling rate, then convolving the charge derivative with the ASIC waveform response template, finally downsampling into 2 MHz.
        '''

        # Check if the asic model is loaded
        if not self.asic_response_flag:
            self.load_asic_response()
        
        preWFLen = len(preSampleTime)
        overSampleTime, overSampleAmp = [], []
        ## If the pre-sampling waveform is too short, just return an empty overSampling waveform.
        if preWFLen < 3:
            overSampleTime = np.zeros(1000)
            overSampleAmp  = np.zeros(1000)
            print("Error: a pre-sampling waveform with length shorter than 3 !!!")
            return

        # Over-sampling of the charge waveform
        sample_time = 0.02 # unit: us, 50 MHz here.
        for TIME_Interpolation in np.arange(preSampleTime[0], preSampleTime[-1], sample_time):
            overSampleTime.append(TIME_Interpolation)
            overSampleAmp.append(np.interp(TIME_Interpolation, preSampleTime, preSampleAmp))
        overWFLen = len(overSampleTime)

        # convolve charge derivative with the CRYO ASIC response
        CRYOAmp_50MHz = np.zeros(overWFLen)
        for i in range(overWFLen-1):
            charge = overSampleAmp[i]
            if i > 0:
                charge = overSampleAmp[i] - overSampleAmp[i-1]
            for j in range(len(self.asic_response_amp)):
                if i+j < overWFLen:
                    CRYOAmp_50MHz[i+j] += charge * self.asic_response_amp[j]

        # Down-sampling to 2 MHz
        downsamplingsize = int(overWFLen/25)
        self.cryoTime, self.cryoAmp = [], []
        for i in range(downsamplingsize-1):
            self.cryoTime.append(overSampleTime[i*25])
            self.cryoAmp.append(CRYOAmp_50MHz[i*25]*54.3)

            # In offline codes, there is a factor 54.3 timed by the amplitude, while I skip it. 
            # Comments there is: 'we actually need to convert noise waveform to ADC, put here before we changed noise lib.'
            # It means the final noise lib should be in unit ADC, while now it still like in e-/us (ENC).
            # Thus, we should keep the signal waveform also in the unit e-/us.
    
    def load_asic_noise(self, entries_per_load=100):
        '''
        The pre-produced noise is saved in this root file with 1e6 entries of filtered noise waveform.
        Read one random entry from the TTree, and cut the waveform to the length.
        '''
        with up.open(self.asic_noise_file) as f:
            tree = f['noiselib']
            total_entries = tree.num_entries
            
            libstart = int(np.random.uniform() * (total_entries - entries_per_load) )
            
            noise_vec_load = tree.arrays(entry_start=libstart, entry_stop=libstart+entries_per_load, library="np")

            self.asic_noises = noise_vec_load['noise_int']



    def generate_noise(self, waveformSize):
        '''
        generate noise amplitudes based on the loaded noise amplitude arrays.
        '''
        if len(self.asic_noises) == 0:
            # If there is no more available noise waveforms in vectors, try loading some new.
            self.load_asic_noise()

        # use the last asic noise array
        noise_vec = self.asic_noises[-1] 
        self.asic_noises = self.asic_noises[::-1]
        
        cut_start = int(np.random.uniform() * (len(noise_vec) - waveformSize))
        self.asic_noise_amp = []
        for i in range(waveformSize):
            self.asic_noise_amp.append(noise_vec[cut_start+i])
        


    def quantization(self, cryoAmp, noiseAmp, Saturation):
        # Comment in offline: 12 bits readout (4096 ticks in total), calculate the number of electrons per tick
        # Saturation is the saturation electron number.
        self.fTruth = np.array(cryoAmp) / (int(Saturation/4096)) 
        self.fTruth = np.round(self.fTruth)
        #print(f'------- Debug output: Noise sum: {np.sum(noiseAmp)}')
        fTempWF = np.zeros(len(noiseAmp))
        for i in range(len(noiseAmp)):
            fTempWF[i] = cryoAmp[i] + noiseAmp[i]

        # 12-bit readout, calculate the electron numebr per tick
        stepsize = int(Saturation / 4096)
        self.outputWF = []
        for elm in fTempWF:
            self.outputWF.append(round(elm/stepsize))
        # Thus, the current waveform output from quantization function has the unit ADC.


    def get_quantized_truthWF(self):
        return self.fTruth

    def diffusion_process(self, Q, x, y, z):
        t_drift = z / self.v_drift # us
        sigmaDiffXY = np.sqrt(2*self.DT*t_drift) / 100. # cm2 -> mm2
        sigmaDiffZ  = np.sqrt(2*self.DL*t_drift) / 100. # cm2 -> mm2
        print(f'Info: transverse diffusion coefficient {sigmaDiffXY} mm, longitudinal diffusion coefficient {sigmaDiffZ} mm in {t_drift} us.')



    def generate_waveform(self, Q, x, y, z):
        self.preTime, self.preAmp = SignalCalculator.ComputeChargeWaveformOnStripWithIons(Q, x, y, z)
        self.convolve_asic_response(self.preTime, self.preAmp)
        self.generate_noise(len(self.cryoAmp))
        self.quantization(self.cryoAmp, self.asic_noise_amp, 40000.)
        