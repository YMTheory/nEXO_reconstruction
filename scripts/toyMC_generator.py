# Script to generate toyMC events based on requirement.
# author: miaoyu@slac.stanford.edu

import numpy as np
import h5py

from toy_digitization import digitization

from IPython import get_ipython
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

if (isnotebook()):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class toyMC_generator():
    def __init__(self) -> None:
        self.filename = None
        self.readme_filename = './toyMC_data/toyMC_generator_readme.txt'
        self.total_entries = 1000
        self.x_xstrip = []
        self.y_xstrip = []
        self.wf_xstrip = []
        self.x_ystrip = []
        self.y_ystrip = []
        self.wf_ystrip = []
        
        self.digi = digitization(SamplingFrequency=2)
    
    def generate_bunches(self, xx, yx, xy, yy, Q0, x0, y0, z0, writeinfo=True):
        self.x_xstrip, self.y_xstrip, self.x_ystrip, self.y_ystrip = xx, yx, xy, yy
        
        for ievt in tqdm(range(self.total_entries)):
            xwf_oneEvt = {}
            ywf_oneEvt = {}
            # generate on strips along x-axis direction:
            for x, y in zip(xx, yx):
                name = f'xstrip_x{x}y{y}'
                self.digi.generate_waveform(Q0, x0-x, y0-y, z0)
                xwf_oneEvt[name] = self.digi.outputWF
        
            # generate on strips along y-axis direction:
            for x, y in zip(xy, yy):
                name = f'ystrip_x{x}y{y}'
                self.digi.generate_waveform(Q0, y0-y, x0-x, z0)
                ywf_oneEvt[name] = self.digi.outputWF

            self.wf_xstrip.append(xwf_oneEvt)
            self.wf_ystrip.append(ywf_oneEvt)
            
        if writeinfo:
            with open(self.readme_filename, 'a') as f:
                # record data file and data info.
                f.write(self.filename)
                f.write('\n')
                f.write(f'Q = {Q0}, ({x0}, {y0}, {z0})\n')
                f.write('Simulated x-strip central positions: \n')
                for x, y in zip(xx, yx):
                    f.write(f'({x}, {y}), ')
                f.write('\n')
                f.write('Simulated y-strip central positions: \n')
                for x, y in zip(xy, yy):
                    f.write(f'({x}, {y}), ')
                f.write('\n')
            
    def save_waveforms(self):
        with h5py.File(self.filename, 'w') as f:
            for ievt in range(self.total_entries):
                groupname = f'event_{ievt}'
                group = f.create_group(groupname)
                for name, xwf in self.wf_xstrip[ievt].items():
                    group.create_dataset(name, data=xwf)
                for name, ywf in self.wf_ystrip[ievt].items():
                    group.create_dataset(name, data=ywf)
                    
        
    
                
        
        