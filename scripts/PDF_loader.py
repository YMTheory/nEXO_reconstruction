import numpy as np
import os

class loader():
    def __init__(self):
        self.grid_diffused_PDFs = None
        self.load_PDF_flag      = False
        self.pdf_length         = 0
        self.gridPDFs_time      = None
        
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

        
        
    def interpolate(self, dX, dY):
        if not self.load_PDF_flag:
            self.load_diffused_PDFs()
            
        # In my current pre-calculated PDFs, there are 16 whole pads on one strips:
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # and the center of the strip is located at the corner between pad7 and pad8.
        # There is a discrepancy between this structure and the real one updated in the offline by me.
        # We shall investigate it later if there is any large influence on this.
        
