import numpy as np
import uproot as up
from array import array
import ROOT
import yaml
from scripts.globals import run_env

class nEXOFieldWP:
    def __init__(self):

        if run_env == 'local':
            ymlfile = '/Users/yumiao/Documents/Works/0nbb/nEXO/Reconstruction/waveform/nEXO_reconstruction/scripts/config.yml'
        elif run_env == 'IHEP':
            ymlfile = '/junofs/users/miaoyu/0nbb/reconstruction/nEXO_reconstruction/scripts/config.yml'
        elif run_env == 'LLNL':
            pass
        elif run_env == "SLAC":
            pass
        else:
            print(f'Error: wrong run environment configuration {run_env}. ')
        
        with open(ymlfile, 'r') as config_file:
            self.filename = yaml.safe_load(config_file)['wp_file']
        
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

        xaxis_edges = array('d', self.xaxis)
        yaxis_edges = array('d', self.yaxis)
        
        self.t_xaxis = ROOT.TAxis(len(self.xaxis)-1, xaxis_edges)
        self.t_yaxis = ROOT.TAxis(len(self.yaxis)-1, yaxis_edges)
        

        