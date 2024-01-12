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
        self.n_step_H = 30
        self.q_cubic = np.zeros((self.n_step_L, self.n_step_L, self.n_step_H))
        
        self.DT = 53.*100/1e6 # mm2/us
        self.DL = 24.8*100/1e6 #mm2/us
        self.v_drift = 1.70 # mm/us

    def diffusion_PDF(self, X0, X): 
        '''
        From zepeng's paper, for N electrons generated at position X0=(x0, y0, z0) at time t0, the electron distribution at X=(x, y, z) and time t is described by a 3-dimensional diffusion equation:
        n(X, t)= N / (8*D_T*sqrt(D_L)[pi*(t-t0)]^{3/2}) * exp{[-(x-x0)^2-(y-y0)^2]/[4*D_T*(t-t0)]} * exp{[-( (z-z0)-v_d(t-t0) )^2]/[4*D_L*(t-t0)]}
        The charge is set as 1 here.
        '''

        x0, y0, z0, t0 = X0
        x, y, z, t = X
        f0 = 1. / (8*self.DT*np.sqrt(self.DL) * (np.pi*np.power((t-t0), 1.5)) ) 
        f1 = np.exp(-((x-x0)**2+(y-y0)**2)/(4*self.DT*(t-t0)))
        f2 = np.exp(-(np.abs(z-z0)-self.v_drift*(t-t0))**2/(4*self.DL*(t-t0)))
        prob = f0 * f1 * f2
        
        return prob
        
           
        
        
    def diffused_point_charges(self):
        v_grid = (self.charge_cubic_L/self.n_step_L)**2 * (self.charge_cubic_H / self.n_step_H)
        for i, xc in tqdm(enumerate(np.linspace(-self.charge_cubic_L/2.+self.x0, self.charge_cubic_L/2.+self.x0, self.n_step_L))):
            for j, yc in enumerate(np.linspace(-self.charge_cubic_L/2.+self.y0, self.charge_cubic_L/2.+self.y0, self.n_step_L)):
                for k, zc in enumerate(np.linspace(-self.charge_cubic_H/2., self.charge_cubic_H/2., self.n_step_H)):
                    tc = np.abs(self.z0 - 0) / self.v_drift
                    X0, X = (self.x0, self.y0, self.z0, 0), (xc, yc, zc, tc)
                    prob_grid = self.diffusion_PDF(X0, X) 
                    self.q_cubic[i, j, k] = prob_grid * self.q0 * v_grid
                    
        # normalization
        self.q_cubic = self.q_cubic * (self.q0 / np.sum(self.q_cubic))

    def induced_currentWF_onStrip(self, strip_x, strip_y, IsAYstrip=True):
        induced_currentWF_onStrip = []
        for i, xc in tqdm(enumerate(np.linspace(-self.charge_cubic_L/2.+self.x0, self.charge_cubic_L/2.+self.x0, self.n_step_L))):
            for j, yc in enumerate(np.linspace(-self.charge_cubic_L/2.+self.y0, self.charge_cubic_L/2.+self.y0, self.n_step_L)):
                for k, zc in enumerate(np.linspace(-self.charge_cubic_H/2., self.charge_cubic_H/2., self.n_step_H)):
                    if self.q_cubic[i, j, k] < 1.0:
                        continue
                    else:
                        if IsAYstrip:
                            x_rel, y_rel = xc - strip_x, yc - strip_y
                        else:
                            x_rel, y_rel = yc - strip_y, xc - strip_x
                        z = self.z0 + zc
                        _, induced_currentWF_onStrip_oneGrid = SignalCalculator.ComputeChargeWaveformOnStripWithIons(self.q_cubic[i, j, k], x_rel, y_rel, z)
                        induced_currentWF_onStrip_oneGrid = np.array(induced_currentWF_onStrip_oneGrid)
                        if len(induced_currentWF_onStrip) == 0:
                            induced_currentWF_onStrip = induced_currentWF_onStrip_oneGrid
                        else:
                            induced_currentWF_onStrip += induced_currentWF_onStrip_oneGrid
        return induced_currentWF_onStrip



    
    def SetSamplingZSeqTemplate(self):
        tmax = 1500
        tmin = 20
        n = 50
        ZBinsVec = [np.exp((1-i/(n+1.)) * np.log(tmax) + i/(n+1) * np.log(tmin) ) for i in range(n)]
        
        # linear spaced fine sampling at short distances
        dZ = ZBinsVec[-2] - ZBinsVec[-1]
        while ZBinsVec[-1] > 1.3 * dZ:
            ZBinsVec.append(ZBinsVec[-1] - dZ)
        ZBinsVec.append(0)
        
        print(ZBinsVec)
            


