import numpy as np
import uproot as up
import ROOT

class nEXOFieldWP:
    def __init__(self):
        self.filename = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/weighting_potentials/singleStripWP6mm_COMSOL.root'
        
        self.xaxis = []
        self.yaxis = []
        self.t_xaxis = None
        self.t_yaxis = None
        
        self.x = None
        self.y = None

        self.hist = None
        
        
    def GetXBin(self, dx):
        return self.t_xaxis.FindBin(dx)
    
    def GetYBin(self, dy):
        return self.t_yaxis.FindBin(dy)
    
    def GetHist(self, xId, yId):
        name = f'wp_x{xId}_y{yId}'
        #print(f'Fetching histogram: {name}')
        f = up.open(self.filename)
        hist = f[name]
        conts, edges = hist.to_numpy()
        cents = (edges[1:] + edges[:-1]) / 2.
        self.x, self.y = cents, conts

        self.hist = ROOT.TH1F(name, "WP histogram", len(edges)-1, edges)
        for i, cont in enumerate(conts):
            self.hist.SetBinContent(i+1, cont)
            


    
    def interpolate(self, z):
        #return np.interp(z, self.x, self.y)
        return self.hist.Interpolate(z)
    
    
    def _load_axes(self):
        f = up.open(self.filename)
        self.xaxis = f['xaxis'].edges()
        self.yaxis = f['yaxis'].edges()

        self.t_xaxis = ROOT.TAxis(len(self.xaxis)-1, self.xaxis[0], self.xaxis[-1])
        self.t_yaxis = ROOT.TAxis(len(self.yaxis)-1, self.yaxis[0], self.yaxis[-1])
        

        