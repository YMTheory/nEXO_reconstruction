from draw import drawer
from loadFits import loader

import numpy as np
import matplotlib.pyplot as plt

ld = loader(inductive=False, noise=True, sscut=1.0, threshold=40, day=20)
d = drawer()

#filename = '/fs/ddn/sdf/group/nexo/users/miaoyu/Reconstruction/Data/Sim/Baseline2019_bb0n_center_seed29_newmap_comsol.nEXOevents.root'
#ld.set_one_file(filename)

n = 1
for i in range(n):

    filename, df_fit_oneevt = ld.get_one_event()
    
    print(f'===== Loading event {i} =====')
    print(df_fit_oneevt)
    
    rootfile = ld.corresponding_rootfile()

    d.config_fitter(rootfile, int(df_fit_oneevt['evtid']),ld.inductive, ld.noise, ld.fine_no, ld.sscut, 0.)
    #d.config_fitter(rootfile, int(df_fit_oneevt['evtid']),ld.inductive, ld.noise, ld.fine_no, ld.sscut, ld.threshold)
    d.config_fitVals(df_fit_oneevt['fitt'], df_fit_oneevt['fitx'], df_fit_oneevt['fity'], df_fit_oneevt['fitQ'], df_fit_oneevt['trueQ'])

    d.fitting_strip_charge()

    #d.draw_channel_waveforms()





