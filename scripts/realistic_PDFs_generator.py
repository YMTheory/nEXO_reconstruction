# This script generates PDFs on adjacent strips with considering the diffusion effects of point charge cluster.

import numpy as np

from scripts.SignalCalculator import SignalCalculator
from scripts.toy_digitization import digitization
from scripts.waveform_WP import waveform_WP

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
    
    def __init__(self, x0=0., y0=0., z0=-1022., q0=1e5, xy_step=1., z_step=1.) -> None:
        self.digi = digitization(SamplingFrequency=2.0)
        
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.q0 = q0
        self.t0 = 0.
        
        self.charge_cubic_L = 10.0 #mm
        self.charge_cubic_H = 6.0  #mm
        self.n_step_L = int(self.charge_cubic_L/xy_step)
        self.n_step_H = int(self.charge_cubic_H/z_step)
        self.q_cubic = np.zeros((self.n_step_L, self.n_step_L, self.n_step_H))
        
        self.DT = 53.*100/1e6 # mm2/us
        self.DL = 24.8*100/1e6 #mm2/us
        self.v_drift = 1.70 # mm/us
        self.sampling_rate = 2.0 # MHz
        self.fAnodeZ = -402.97 #mm

        self.wp_gen = waveform_WP()
        self.wp_gen.initialize()
        self.wp_gen.initialize_PCDZ(np.arange(self.z0-self.charge_cubic_H/2., self.z0+self.charge_cubic_H/2., z_step))
        
        self.strip_charge_time = None
        self.strip_charge_waveform = None
        self.strip_quantized_current_time = None
        self.strip_quantized_current_waveform = None

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
                for k, zc in enumerate(np.linspace(-self.charge_cubic_H/2.+self.fAnodeZ, self.charge_cubic_H/2.+self.fAnodeZ, self.n_step_H)):
                    tc = np.abs(self.fAnodeZ - self.z0) / self.v_drift
                    X0, X = (self.x0, self.y0, self.z0, 0), (xc, yc, zc, tc)
                    prob_grid = self.diffusion_PDF(X0, X) 
                    self.q_cubic[i, j, k] = prob_grid * self.q0 * v_grid
                    
        # normalization
        self.q_cubic = self.q_cubic * (self.q0 / np.sum(self.q_cubic))
        print(f"-> Smearing charge density in three-dimension: sigma_xy = {self.DL} mm2/us and sigma_z = {self.DT} mm2/us.")

    def induced_currentWF_onStrip(self, strip_x, strip_y, IsAXstrip=True):
        minZ, maxZ = self.z0 - self.charge_cubic_H/2., self.z0 + self.charge_cubic_H/2.
        self.wp_gen.DetermineSamplingSequence(maxZ, minZ)
        
        self.diffused_point_charges()
        
        induced_time_onStrip = 0.
        induced_chargeWF_onStrip = []
        
        for i, xc in tqdm(enumerate(np.linspace(-self.charge_cubic_L/2.+self.x0, self.charge_cubic_L/2.+self.x0, self.n_step_L))):
            for j, yc in enumerate(np.linspace(-self.charge_cubic_L/2.+self.y0, self.charge_cubic_L/2.+self.y0, self.n_step_L)):
                for k, zc in enumerate(np.linspace(-self.charge_cubic_H/2., self.charge_cubic_H/2., self.n_step_H)):
                    xc = self.x0 + xc
                    yc = self.y0 + yc
                    z = self.fAnodeZ - (self.z0 + zc)
                    if self.q_cubic[i, j, k] < 1.0:
                        continue
                    else:
                        if IsAXstrip:
                            x_rel, y_rel = xc - strip_x, yc - strip_y
                        else:
                            x_rel, y_rel = yc - strip_y, xc - strip_x
                        x_rel, y_rel = np.abs(x_rel), np.abs(y_rel)
                        dx_a, dy_a = int(x_rel/6.0), int(y_rel/6.0)
                        #_, induced_currentWF_onStrip_oneGrid = SignalCalculator.ComputeChargeWaveformOnStripWithIons(self.q_cubic[i, j, k], x_rel, y_rel, z)
                        self.wp_gen.CalcPointChargeWaveformOnChannel(x_rel, y_rel, z, self.q_cubic[i, j, k])
                        inducde_time_onStrip_oneGrid     = self.wp_gen.onechannel_time_pointcharge
                        induced_chargeWF_onStrip_oneGrid = self.wp_gen.onechannel_wf_pointcharge
                        if len(induced_chargeWF_onStrip_oneGrid) == 0:
                            continue
                        else:
                            induced_time_onStrip = inducde_time_onStrip_oneGrid
                            if len(induced_chargeWF_onStrip) == 0:
                                induced_chargeWF_onStrip = np.array(induced_chargeWF_onStrip_oneGrid)
                            else:
                                induced_chargeWF_onStrip += induced_chargeWF_onStrip_oneGrid

        self.strip_charge_time = np.array(induced_time_onStrip)
        self.strip_charge_waveform = np.array(induced_chargeWF_onStrip)
        

    def quantize_waveform(self):
        self.digi.convolve_asic_response(self.strip_charge_time, self.strip_charge_waveform)
        self.digi.quantization_trueWF(self.digi.cryoAmp, 40000.)
        self.strip_quantized_current_waveform = self.digi.fTruth
        self.strip_quantized_current_time = np.arange(0, len(self.strip_quantized_current_waveform), 1) * 0.5