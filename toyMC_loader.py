# Svecript to load toyMC waveforms from h5 files.

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import re

from toy_digitization import digitization

class toyMC_loader():
    def __init__(self) -> None:
        self.toyMC_filename = None
        self.waveform_time = None
        self.waveform_oneEvt = None
        self.waveform_multiChannel_oneEvt = None
        
        self.digi = digitization(SamplingFrequency=2.0)
        
    def load_h5file(self, evtNo):
        with h5.File(self.toyMC_filename, 'r') as f:
            dset_name = 'waveform_' + str(evtNo)
            dset = f[dset_name]
            
            self.waveform_oneEvt = []
            for elem in dset:
                self.waveform_oneEvt.append(elem)
                
        self.waveform_time = np.arange(len(self.waveform_oneEvt)) * 0.5
                
    def display_event(self, evtNo, Q, x, y, z, noiseFlag=False):
        self.load_h5file(evtNo)
        
        self.digi.generate_waveform(Q, x, y, z)

        _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.waveform_time, self.waveform_oneEvt, 'o-', ms=5, lw=2, label='from dataset')

        if noiseFlag:
            ax.plot(self.digi.cryoTime, self.digi.outputWF, ':', lw=2, label='from toyMC')
        else:
            ax.plot(self.digi.cryoTime, self.digi.fTruth, ':', lw=2, label='from toyMC')

        
        ax.set_xlabel('Drift time [us]', fontsize=12)
        ax.set_ylabel('adc', fontsize=12)
        ax.legend(prop={'size':12})
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlim(self.waveform_time[-70], self.waveform_time[-15])
        plt.tight_layout()
        
    
    def load_h5file_multiChannels(self, evtNo):
        with h5.File(self.toyMC_filename, 'r') as f:
            groupname = f'event_{evtNo}'
            try:
                group = f[groupname]
                self.waveform_multiChannel_oneEvt = {}
            
                for dsetname in group.keys():
                    dset = group[dsetname]
                    wf = []
                    for elem in dset:
                        wf.append(elem)
                    self.waveform_multiChannel_oneEvt[dsetname] = wf
                return True
            except:
                print(f'Error when loading group {groupname}.')
                return False
                
    def display_event_multiChannels(self, evtNo, ):
        self.load_h5file_multiChannels(evtNo)
        
        _, ax = plt.subplots(figsize=(6, 4))
        for name, wf in self.waveform_multiChannel_oneEvt.items():
            times = np.arange(len(wf)) * 0.5
            ax.plot(times, wf, alpha=0.65, label=name)
        
        ax.set_xlabel('Drift time [us]', fontsize=12)
        ax.set_ylabel('adc', fontsize=12)
        ax.legend(loc='upper left', prop={'size':12})
        ax.tick_params(axis='both', labelsize=12)
        #ax.set_xlim(self.waveform_time[-70], self.waveform_time[-15])
        plt.tight_layout()


    def assembling_for_fitter(self):
        times, wfs, sxs, sys, ystrips = [], [], [], [], []
        for name, wf in self.waveform_multiChannel_oneEvt.items():
            pattern = r"-?\d+\.?\d*"
            sx, sy = re.findall(pattern, name)
            sxs.append(float(sx))
            sys.append(float(sy))
            if name.startswith('xstrip'):
                ystrips.append(False)
            else:
                ystrips.append(True)
            
            times.append(np.arange(len(wf))*0.5)
            wfs.append(wf)
            
        return times, wfs, sxs, sys, ystrips