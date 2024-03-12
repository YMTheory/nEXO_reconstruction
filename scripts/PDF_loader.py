import numpy as np
import os
from scipy.interpolate import griddata

class loader():
    def __init__(self, mode='PCD', charge_x=0.0, charge_y=0.0, charge_z=-622.0, charge_q=1.0e5):
        
        self.charge_x           = charge_x
        self.charge_y           = charge_y 
        self.charge_z           = charge_z
        self.charge_q           = charge_q
        
        self.grid_diffused_PDFs = None
        self.pcd_diffused_PDFs  = None
        self.load_PDF_flag      = False
        self.pdf_length         = 0
        self.gridPDFs_time      = None
        self.pcdPDFs_time       = None

        self.load_mode = mode
        
    def load_diffused_PDFs(self):
        self.grid_diffused_PDFs = {} # dictionary to store the diffused PDFs.
        for x in range(0, 54, 1):
            for y in range(0, 31, 1):
                if x <= 10:
                    filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/PDFs/diffusionPDFs_YstripatX0.0Y0.0_chargeX{x+12:.1f}Y{y:.1f}Z619.63Q100000.0.npy' 
                ###########################
                else:
                    filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/PDFs/diffusionPDFs_YstripatX0.0Y0.0_chargeX{x:.1f}Y{y:.1f}Z619.63Q100000.0.npy' 
                if not os.path.exists(filename):
                    print(f'Error: {filename} does not exists!' )
                    continue
                else:
                    arr = np.load(filename)
                    self.gridPDFs_time = arr[0]
                    
                    # hard-coded currently, modification required later:
                    ## calculate the charge attenuation during drift.
                    drift_time = 619.63 / self.v_drift # us
                    att = np.exp(-drift_time/self.lifetime)
                    
                    self.gridPDFs_diffusion[f'X{x}Y{y}'] = arr[1] * att
        self.pdf_length = len(self.gridPDFs_time)
        self.load_PDF_flag = True
        print('The diffusion PDFs loaded successfully!')

    
    def load_diffusedPCD_PDFs(self):
        self.pcd_diffused_PDFs = {}
        for x in np.arange(0, 6.5, 0.5):
            for y in np.arange(0, 6.5, 0.5):
                filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/diffPDFs/z-622mm/stencilPDF_xstripx{x:.1f}y{y:.1f}.npz'
                if not os.path.exists(filename):
                    print(f'Error: {filename} does not exists!' )
                    continue
                else:
                    arr = np.load(filename)
                    dict_name = f'dx{x:.1f}dy{y:.1f}'
                    self.pcd_diffused_PDFs[dict_name] = arr
                    self.pdf_length = len(arr['time'])
        self.load_PDF_flag = True
        print('The diffusion PCD PDFs loaded successfully!')
    
    def get_one_stencil(self, name, t0):
        if name not in self.pcd_diffused_PDFs:
            print(f'Error: {name} not in the pre-loaded dictionary.')
            return 0.
        else:
            t, wf = self.pcd_diffused_PDFs[name]['time'], self.pcd_diffused_PDFs[name]['wf']
            return np.interp(t0, t, wf)
        
        
    def interpolate(self, dX, dY):
        if not self.load_PDF_flag:
            if self.load_mode == 'grid':
                self.load_diffused_PDFs()
            elif self.load_mode == 'PCD':
                self.load_diffusedPCD_PDFs()
            else:
                print(f'Error: unknown load mode {self.load_mode}!')

        else:
            print('The diffusion PDFs have already been pre-loaded!')
            
        xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
        if dX < xmin or dX > xmax or dY < ymin or dY > ymax:
            print(f'Error: dX or dY out of range! ({dX}, {dY})')
            return np.zeros(self.pdf_length)
        else:
            step = 0.5
            x_left = int(np.abs(dX) / step) * step
            x_right = x_left + step
            y_down = int(np.abs(dY) / step) * step
            y_up = y_down + step
            name00 = f'dx{x_left:.1f}dy{y_down:.1f}'
            name01 = f'dx{x_left:.1f}dy{y_up:.1f}'
            name10 = f'dx{x_right:.1f}dy{y_down:.1f}'
            name11 = f'dx{x_right:.1f}dy{y_up:.1f}'
            
            if name00 not in self.pcd_diffused_PDFs :
                print(f'Error: {name00} not in the pre-loaded dictionary.')
                return np.zeros(self.pdf_length)
            else:
                f00 = self.pcd_diffused_PDFs[name00]['wf']
            if name01 not in self.pcd_diffused_PDFs :
                print(f'Error: {name01} not in the pre-loaded dictionary.')
                return np.zeros(self.pdf_length)
            else:
                f01 = self.pcd_diffused_PDFs[name01]['wf']
            if name10 not in self.pcd_diffused_PDFs :
                print(f'Error: {name10} not in the pre-loaded dictionary.')
                return np.zeros(self.pdf_length)
            else:
                f10 = self.pcd_diffused_PDFs[name10]['wf']
            if name11 not in self.pcd_diffused_PDFs :
                print(f'Error: {name11} not in the pre-loaded dictionary.')
                return np.zeros(self.pdf_length)
            else:
                self.pcdPDFs_time = self.pcd_diffused_PDFs[name11]['time']
                f11 = self.pcd_diffused_PDFs[name11]['wf']

            Xs = np.array([x_left, x_left, x_right, x_right])
            Ys = np.array([y_down, y_up, y_down, y_up])
            Zs = np.array([f00, f01, f10, f11])
            f = griddata((Xs, Ys), Zs, (np.abs(dX), np.abs(dY)), method='linear')
            
            return f

            
    def diffused_waveform_oneChannel(self, strip_x, strip_y, IsAXstrip, t):
        dX, dY = strip_x - self.charge_x, strip_y - self.charge_y
        if not IsAXstrip:
            dY, dX = strip_x - self.charge_x, strip_y - self.charge_y

        f = self.interpolate(dX, dY)
        return np.interp(t, self.pcdPDFs_time, f)
