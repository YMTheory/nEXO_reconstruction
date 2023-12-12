# Generating PDF templates for fitting
# Author: miaoyu@slac.stanford.edu

import numpy as np
import h5py as h5
#import matplotlib.pyplot as plt

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

class pdf_generator():
    def __init__(self) -> None:
        self.X = []
        self.Y = []
        self.filename = None
        
        self.vertex_x, self.vertex_y = [], []

        self.digi = digitization(SamplingFrequency=2.0)
        
        
    def _set_filename(self, name):
        self.filename = name
        
    def _is_point_inTriangle(self, p, t):
        # p: point coordinate (px, py)
        # t: vertice coordinate [(x1, y1), (x2, y2), (x3, y3)]
        
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, t[0], t[1])
        d2 = sign(p, t[1], t[2])
        d3 = sign(p, t[2], t[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)
        

    def _is_point_onStrip(self, p, numPads=16, padSize=6.):
        self.vertex_x = []
        self.vertex_y = []
        
        self.vertex_x.append(0)
        self.vertex_y.append(0)

        for i in range(numPads):
            self.vertex_x.append(padSize/2. * (i+1))
            if i % 2:
                self.vertex_y.append(0.)
            else:
                self.vertex_y.append(padSize/2.)

        px, py = p[0], p[1]
        if py >= 3:
            return False
        else:
            flag = False
            for padid in range(int(numPads/2)):
                flag = self._is_point_inTriangle((px, py),[(self.vertex_x[0+padid*2], self.vertex_y[0+padid*2]), (self.vertex_x[1+padid*2], self.vertex_y[1+padid*2]), (self.vertex_x[2+padid*2], self.vertex_y[2+padid*2])] )
                if flag:
                    return True
            return False
    
        
        
    def _generate_coordinates(self, numPads=16, padSize=6):
        
        self.X, self.Y = [], []
        
        # Generating points on the strip edge:
        self.X.append(0)
        self.Y.append(0)
        self.vertex_x.append(0)
        self.vertex_y.append(0)
        for i in range(numPads):
            self.X.append(padSize/2. * (i+1) )
            if i%2:
                self.Y.append(0.)
            else:
                self.Y.append(padSize/2.)
            self.vertex_x.append(self.X[-1])
            self.vertex_y.append(self.Y[-1])
        # Generating points outside the strip:
        xs = [0, 10, 20, 40, 48]
        step_x = [0.5, 1, 2, 4]
        ys = [0, 10, 20, 30]
        step_y = [0.5, 1, 2]
        for i in range(len(step_x)):
            num_x = int((xs[i+1]-xs[i]) / step_x[i])
            for j in range(len(step_y)):
                num_y = int((ys[j+1]-ys[j]) / step_y[j])

                for ix in range(num_x):
                    tmp_x = xs[i] + step_x[i] * ix
                    for iy in range(num_y):
                        tmp_y = ys[j] + step_y[j] * iy
                        
                        # if the point is on the Strip
                        flag = False
                        if tmp_y < 3.0:
                            for padid in range(8):
                                flag = self._is_point_inTriangle((tmp_x, tmp_y), [(self.vertex_x[0+padid*2], self.vertex_y[0+padid*2]), (self.vertex_x[1+padid*2], self.vertex_y[1+padid*2]), (self.vertex_x[2+padid*2], self.vertex_y[2+padid*2])])
                                if flag:
                                    break
                        if not flag:
                            self.X.append(tmp_x)
                            self.Y.append(tmp_y)
        print(f'Total generated {len(self.X)} points for PDF templates.')


    def _generate_PDFs(self, q, z):
        self._generate_coordinates()
        
        induction_pdfs = {}
        #for x, y in tqdm(zip(self.X, self.Y)):
        for i in tqdm(range(len(self.X))):
            x, y = self.X[i], self.Y[i]
            pdf_name = f'x{x:.1f}y{y:.1f}'
            tt, wf = SignalCalculator.ComputeChargeWaveformOnStripWithIons(q, x, y, z)
            self.digi.convolve_asic_response(tt, wf)
            self.digi.generate_noise(len(self.digi.cryoAmp))
            self.digi.quantization(self.digi.cryoAmp, self.digi.asic_noise_amp, 40000.)
            pdf = self.digi.get_quantized_truthWF()
            induction_pdfs[pdf_name] = pdf
            
        with h5.File(self.filename, 'w') as f:
            for name, pdf in induction_pdfs.items():
                f.create_dataset(name, data=pdf)
