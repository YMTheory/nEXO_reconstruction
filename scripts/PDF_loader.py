import numpy as np
import os
from scipy.interpolate import griddata
import yaml
import copy
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

from scripts.globals import run_env
class loader():
    def __init__(self, mode='PCD' ):
        
        self.grid_diffused_PDFs = None
        self.pcd_diffused_PDFs  = None
        self.load_PDF_flag      = False
        self.pdf_length         = 0
        self.gridPDFs_time      = None
        self.pcdPDFs_time       = None
        self.PDF_fine           = 0   # 0 for coarse, 1 for fine

        self.verbose            = False

        self.path = None
        if run_env == 'LOCAL':
            ymlfile = '/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/scripts/config.yml'
        elif run_env == 'IHEP':
            ymlfile = '/junofs/users/miaoyu/0nbb/reconstruction/nEXO_reconstruction/scripts/config_IHEP.yml'
        elif run_env == 'LLNL':
            pass
        elif run_env == "SLAC":
            ymlfile = '/fs/ddn/sdf/group/nexo/users/miaoyu/Reconstruction/Softwares/nEXO_reconstruction/scripts/config_SLAC.yml'
        else:
            print(f'Error: wrong run environment configuration {run_env}. ')
        with open(ymlfile, 'r' ) as config_file:
            
            filelist = yaml.safe_load(config_file)
            self.path = filelist['pdf_path']

        self.load_mode = mode

    def _set_verbose(self, vb):
        self.verbose = vb
        
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
        #for x in np.arange(0, 30, 0.5):
        #    for y in np.arange(0, 15, 0.5):
        #        #filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/diffPDFs/z-622mm_IHEP/stencilPDF_xstripx{x:.1f}y{y:.1f}.npz'
        #        #filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/diffPDFs/z-622mm/stencilPDF_xstripx{x:.1f}y{y:.1f}.npz'
        #        filename = f'{self.path}stencilPDF_chargex{x:.1f}y{y:.1f}z-622.0_xstripx0.0y0.0.npz'
        #        if not os.path.exists(filename):
        #            print(f'Error: {filename} does not exists!' )
        #            continue
        #        else:
        #            with np.load(filename) as f:
        #            #arr = np.load(filename)
        #                tmp = copy.copy(f)
        #                time = tmp['time']
        #                wf = tmp['wf']
        #                arr = np.vstack((time, wf))
        #                self.pdf_length = len(time)
        #            dict_name = f'dx{x:.1f}dy{y:.1f}'
        #            self.pcd_diffused_PDFs[dict_name] = arr
        # finer PDFs in 6mm * 6mm zone
        print(f'=====> Loading diffusion PCD PDF files from {self.path} <<< ')
        if self.PDF_fine == 0:
            # coarse PDFs
            pdf_xmin, pdf_xmax, pdf_xstep = 0, 30, 0.5
            pdf_ymin, pdf_ymax, pdf_ystep = 0, 6.5, 0.5
        elif self.PDF_fine == 1:
            # fine PDFs
            pdf_xmin, pdf_xmax, pdf_xstep = 0, 24, 0.1
            pdf_ymin, pdf_ymax, pdf_ystep = 0, 6.1, 0.1
        for x in tqdm(np.arange(pdf_xmin, pdf_xmax, pdf_xstep)):
            for y in np.arange(pdf_ymin, pdf_ymax, pdf_ystep):
                #filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/diffPDFs/z-622mm_IHEP/stencilPDF_xstripx{x:.1f}y{y:.1f}.npz'
                #filename = f'/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/diffPDFs/z-622mm/stencilPDF_xstripx{x:.1f}y{y:.1f}.npz'
                filename = f'{self.path}stencilPDF_chargex{x:.1f}y{y:.1f}z-622.0_xstripx0.0y0.0.npz'
                if not os.path.exists(filename):
                    print(f'Error: {filename} does not exists!' )
                    continue
                else:
                    with np.load(filename) as f:
                    #arr = np.load(filename)
                        tmp = copy.copy(f)
                        time = tmp['time']
                        wf = tmp['wf']
                        arr = np.vstack((time, wf))
                        self.pdf_length = len(time)
                    dict_name = f'dx{x:.1f}dy{y:.1f}'
                    self.pcd_diffused_PDFs[dict_name] = arr
        self.load_PDF_flag = True
        print('The diffusion PCD PDFs loaded successfully!')
    
    def get_one_stencil(self, name, t0):
        if name not in self.pcd_diffused_PDFs:
            print(f'Error: {name} not in the pre-loaded dictionary.')
            return 0.
        else:
            t, wf = self.pcd_diffused_PDFs[name][0, :], self.pcd_diffused_PDFs[name][:, 1]
            return np.interp(t0, t, wf)
        
        
    def interpolate(self, dX, dY):
        '''
        The dX and dY are the relative distance between the point charge center (before diffusion) and the strip center.
        Here the y-strip should already be rotated to the x direction with the correct rotated dX and dY.
        '''
        if not self.load_PDF_flag:
            if self.load_mode == 'grid':
                self.load_diffused_PDFs()
            elif self.load_mode == 'PCD':
                self.load_diffusedPCD_PDFs()
            else:
                print(f'Error: unknown load mode {self.load_mode}!')

        #else:
        #    print('The diffusion PDFs have already been pre-loaded!')
            
        if self.PDF_fine == 0:
            xmin, xmax, ymin, ymax = -29.5, 29.5, -6., 6.
        elif self.PDF_fine == 1:
            xmin, xmax, ymin, ymax = -23.8, 23.8, -6., 6.
        PadSize = 6.0
        nPadHalfStrip = 8
        
        if dX <= xmin or dX >= xmax:
            # The point charge is too far away from the strip along the x-axis (which is not the alignment direction of the strip), omit the contributions from this charge then.
            print(f"The point charge is too far away from the strip along the x-axis! ({dX:.2f}, {dY:.2f}) mm.")
            self.pcdPDFs_time = np.arange(0, self.pdf_length, 1)
            return np.zeros(self.pdf_length) # return a zero waveform, no contributions from this charge on the strip.
        
        #elif dY < ymin or dY > ymax and np.abs(dY) <= PadSize * nPadHalfStrip:
            
        if np.abs(dY) > PadSize * nPadHalfStrip:
            print(f"The point charge is out the range along the strip direction! ({dX:.2f}, {dY:.2f}) mm.")
            # Currently I set them as 0 but it could be incorrect if the charge is still close to the end of the strip.
            self.pcdPDFs_time = np.arange(0, self.pdf_length, 1)
            return np.zeros(self.pdf_length) # return a zero waveform, no contributions from this charge on the strip.
        
        if self.verbose:
            print(f'Replace dY ({dY:.2f}) by {dY%PadSize:.2f} due to symmetry.')
        dY = dY % PadSize

        if self.PDF_fine == 0:
            step = 0.5
        elif self.PDF_fine == 1 :
            step = 0.1
        x_left = int(np.abs(dX) / step) * step
        x_right = x_left + step
        y_down = int(np.abs(dY) / step) * step
        y_up = y_down + step
        name00 = f'dx{x_left:.1f}dy{y_down:.1f}'
        name01 = f'dx{x_left:.1f}dy{y_up:.1f}'
        name10 = f'dx{x_right:.1f}dy{y_down:.1f}'
        name11 = f'dx{x_right:.1f}dy{y_up:.1f}'
        if self.verbose:
            print(f'Interpolation position ({dX:.2f}, {dY:.2f}) with corner pdf names [{name00}, {name01}, {name10}, {name11}].')
        
        if name00 not in self.pcd_diffused_PDFs :
            print(f'Error: {name00} not in the pre-loaded dictionary.')
            return np.zeros(self.pdf_length)
        else:
            f00 = self.pcd_diffused_PDFs[name00][1, :]
        if name01 not in self.pcd_diffused_PDFs :
            print(f'Error: {name01} not in the pre-loaded dictionary.')
            return np.zeros(self.pdf_length)
        else:
            f01 = self.pcd_diffused_PDFs[name01][1, :]
        if name10 not in self.pcd_diffused_PDFs :
            print(f'Error: {name10} not in the pre-loaded dictionary.')
            return np.zeros(self.pdf_length)
        else:
            f10 = self.pcd_diffused_PDFs[name10][1, :]
        if name11 not in self.pcd_diffused_PDFs :
            print(f'Error: {name11} not in the pre-loaded dictionary.')
            return np.zeros(self.pdf_length)
        else:
            self.pcdPDFs_time = self.pcd_diffused_PDFs[name11][0, :]
            f11 = self.pcd_diffused_PDFs[name11][1, :]

        Xs = np.array([x_left, x_left, x_right, x_right])
        Ys = np.array([y_down, y_up, y_down, y_up])
        Zs = np.array([f00, f01, f10, f11])
        f = griddata((Xs, Ys), Zs, (np.abs(dX), np.abs(dY)), method='linear')
        
        return f

            
    def diffused_waveform_oneChannel(self, dX, dY, t):
        '''
        The dX and dY are the relative distance between the point charge center (before diffusion) and the strip center.
        Here the y-strip should already be rotated to the x direction with the correct rotated dX and dY.
        '''
        f = self.interpolate(dX, dY)
        return np.interp(t, self.pcdPDFs_time, f)
    
    
