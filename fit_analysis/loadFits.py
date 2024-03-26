import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob
import os
import re

class loader():
    def __init__(self, inductive=True, noise=True, fine='fine', sscut=1.0, threshold=40, rdminit=True, year=2024, month=3, day=22, label='default'):
        self.inductive  = inductive
        self.noise      = noise
        self.fine       = fine
        if self.fine == 'fine':
            self.fine_no = 1
        else:
            self.fine_no = 0
        self.sscut      = sscut
        self.threshold  = threshold
        self.rdminit    = rdminit
        self.year       = year
        self.month      = month
        self.day        = day
        self.label      = label

        self.onefile    = None
        self.rootfile   = None

    def _get_label(self):
        return self.label
    

    ###### format the input filename:
    def format_filenames(self):
        path = '/fs/ddn/sdf/group/nexo/users/miaoyu/Reconstruction/Data/Fits/results'
        self.prefix = 'fit_MCtruthSS_seed'
        #suffix = '_inductive'+str(self.inductive)+'_noise'+str(self.noise)+'_'+self.fine+'newPDFs_'+str(zero)+str(self.month)+str(self.day)+'_refitmax5_SScut'+f'{self.sscut:.1f}'+'mm_ampthre'+f'{self.threshold}'+'.csv'
        self.suffix = '_inductive'+str(self.inductive)+'_noise'+str(self.noise)+'_'+self.fine+'newPDFs_'+'refitmax5_SScut'+f'{self.sscut:.1f}'+'mm_ampthre'+f'{self.threshold}_rdminit'+str(self.rdminit) + '_'+str(self.year) + str(self.month) + str(self.day) + '.csv'
        
        
        # Match all files, do not specify the seed manually
        pattern = os.path.join(path, "*"+self.suffix)
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
            print('--> ' + str(self.suffix))




    def load_all_fitVals(self, matching_files):
        varnamelist = ['evtid', 'status', 'chi2', 'fitt', 'fitterr', 'fitx', 'fitxerr', 'fity', 'fityerr', 'fitQ' , 'fitQerr', 'trueQ', 'itgQ']
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


    def successful_fitNumber(self):
        df_fit = self.load()
        successful_num = len(df_fit[df_fit['status'] == 1] )
        tot_num = df_fit.shape[0]
        print(f'>>> Configure {self.label}: total fitting event is {tot_num} among which {successful_num} is successful.')



    def set_one_file(self, filename):
        self.onefile = filename

    def get_one_event(self):
        if self.onefile == None:
            matching_files = self.format_filenames()
            self.onefile = np.random.choice(matching_files)

        df_fit_onefile = self.load_all_fitVals([self.onefile])
        df_fit_oneevt  = df_fit_onefile.sample()

        
        return self.onefile, df_fit_oneevt.iloc[0]


    def corresponding_rootfile(self):
        rootfile_path = '/fs/ddn/sdf/group/nexo/users/miaoyu/Reconstruction/Data/Sim/'
        try:
            pattern = re.compile(r'seed(\d+)')
            match = pattern.search(self.onefile)
            if match:
                seed_number = match.group(1)
                self.rootfile = rootfile_path + "Baseline2019_bb0n_center_seed"+seed_number+"_newmap_comsol.nEXOevents.root"
        except Exception as err:
            print(type(err))

        return self.rootfile
                






