import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob
import os

class loader():
    def __init__(self, inductive, noise, fine, sscut, threshold, year, month, day, label):
        self.inductive  = inductive
        self.noise      = noise
        self.fine       = fine
        self.sscut      = sscut
        self.threshold  = threshold
        self.year       = year
        self.month      = month
        self.day        = day
        self.label      = label


    def _get_label(self):
        return self.label
    

    ###### format the input filename:
    def format_filenames(self):
        path = '/fs/ddn/sdf/group/nexo/users/miaoyu/Reconstruction/Data/Fits/results'
        prefix = 'fit_MCtruthSS_seed'
        suffix = '_inductive'+self.inductive+'_noise'+self.noise+'_'+self.fine+'newPDFs_'+self.month+self.day+'_refitmax5_SScut'+self.sscut+'mm_ampthre'+self.threshold+'.csv'
        
        
        # Match all files, do not specify the seed manually
        pattern = os.path.join(path, "*"+suffix)
        matching_files = glob.glob(pattern)

        return matching_files
    
    
    def get_fitVals(self, filename, var):
        df = pd.df = pd.read_csv(filename)
        return df[var]
    
    
    def merge_fitVals(self, filelist, var):
        var_list = []
        for filename in filelist:
            var_list.append(self.get_fitVals(filename, var))
        return np.concatenate(var_list)
    
    
    def load_one_fitVals(self, varname):
        matching_files = self.format_filenames()
        try:
            vararray = self.merge_fitVals(matching_files, varname)
            return vararray
        except ValueError:
            print('ValueError: there is no matching files for this fitting configuration.')




    def load_all_fitVals(self, matching_files):
        varnamelist = ['evtid', 'status', 'chi2', 'fitt', 'fitterr', 'fitx', 'fitxerr', 'fity', 'fityerr', 'fitQ' , 'fitQerr', 'trueQ',]
        varlist = []
    
        for varname in varnamelist:
            varlist.append(     self.merge_fitVals(matching_files, varname)     )
    
        varlist = np.array(varlist) 
    
        df_fit = pd.DataFrame({name : np.array(val) for name, val in zip(varnamelist, varlist)} )
    
    
        return df_fit
    
    
    
    def load(self):
    
        matching_files = self.format_filenames()
    
        df_fit = self.load_all_fitVals(matching_files)
    
        return df_fit
    
    #main()












