# This script generates PDFs on adjacent strips with considering the diffusion effects of point charge cluster.

import numpy as np

from SignalCalculator import SignalCalculator
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

class generator():
    
    def __init__(self) -> None:
        self.digi = digitization(SamplingFrequency=2.0)
        
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0
        self.q0 = 1.0
        self.t0 = 0
        
        self.charge_cubic_L = 10.0 #mm
        self.charge_cubic_H = 6.0  #mm
        self.n_step_L = 50
        self.n_step_H = 50
        
    def calculate_induced_charge_on_single_strip_atTime(self, x_strip, y_strip, ystrip_flag, t):
        
        z_center = self.z0 + self.digi.v_drift * (t-self.t0)
        charge_induced_onstrip_atTime = 0.
        for xc in tqdm(np.linspace(-self.charge_cubic_L/2.+self.x0, self.charge_cubic_L/2.+self.x0, self.n_step_L)):
            for yc in np.linspace(-self.charge_cubic_L/2.+self.y0, self.charge_cubic_L/2.+self.y0, self.n_step_L):
                for zc in np.linspace(-self.charge_cubic_H/2.+z_center, self.charge_cubic_H/2.+z_center, self.n_step_H):
                    q_grid = self.digi.diffused_PDF(self.q0, (self.x0, self.y0, self.z0, self.t0), (xc, yc, zc, t))
                    if ystrip_flag:
                        charge_induced_bygrid_onstrip_atTime = SignalCalculator.InducedChargeNEXOStrip(q_grid, xc-x_strip, yc-y_strip, zc)
                    else:
                        charge_induced_bygrid_onstrip_atTime = SignalCalculator.InducedChargeNEXOStrip(q_grid, yc-y_strip, xc-x_strip, zc)
                    
                    charge_induced_onstrip_atTime += charge_induced_bygrid_onstrip_atTime
        return charge_induced_onstrip_atTime
    
    


