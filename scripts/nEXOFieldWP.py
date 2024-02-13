import numpy as np
import uproot as up
#import ROOT

class nEXOFieldWP:
    def __init__(self):
        self.filename = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/singleStripWP6mm_COMSOL.root'
        
        self.xaxis = []
        self.yaxis = []
        
        self.x = None
        self.y = None
        
        
    def GetXBin(self, dx):
        findId = np.searchsorted(self.xaxis, dx, side='right') 
        if findId == 1:
            return findId
        else:
            return findId - 1
    
    def GetYBin(self, dy):
        findId = np.searchsorted(self.yaxis, dy, side='right') 
        if findId == 1:
            return findId
        else:
            return findId - 1
    
    def GetHist(self, xId, yId):
        name = f'wp_x{xId}_y{yId}'
        print(f'Fetching histogram: {name}')
        f = up.open(self.filename)
        hist = f[name]
        conts, edges = hist.to_numpy()
        cents = (edges[1:] + edges[:-1]) / 2.
        self.x, self.y = cents, conts

    
    def interpolate(self, z):
        return np.interp(z, self.x, self.y)
    
    
    def _load_axes(self):
        f = up.open(self.filename)
        self.xaxis = f['xaxis'].edges()
        self.yaxis = f['yaxis'].edges()
        

        