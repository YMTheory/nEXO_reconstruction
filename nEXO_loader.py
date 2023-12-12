import uproot as up
import numpy as np

import matplotlib.pyplot as plt

class loader():
    def __init__(self) -> None:
        self.filename = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/Baseline2019_bb0n_FullLXe_seed52_comsol.nEXOevents.root'
        self.load_nentries = 100
        self.start_evt = 0

        self.fWFAndNoise    = None
        self.fTileX         = None
        self.fTileY         = None
        self.fLocalID       = None
        self.fXPosition     = None
        self.fYPosition     = None
        self.fCharge        = None
        
    def _load_event(self):
        with up.open(self.filename) as f:
            tree = f['Event/Elec/ElecEvent']
            #print(tree.keys())
            self.fWFAndNoise    = tree['ElecEvent/fElecChannels/fElecChannels.fWFAndNoise'].array(entry_start=0, entry_stop=10)
            self.fXPosition     = tree['ElecEvent/fElecChannels/fElecChannels.fXPosition'].array(entry_start=0, entry_stop=10)
            self.fYPosition     = tree['ElecEvent/fElecChannels/fElecChannels.fYPosition'].array(entry_start=0, entry_stop=10)
            self.fCharge        = tree['ElecEvent/fElecChannels/fElecChannels.fChannelCharge'].array(entry_start=0, entry_stop=10)
            self.fLocalID       = tree['ElecEvent/fElecChannels/fElecChannels.fChannelLocalId'].array(entry_start=0, entry_stop=10)
            self.fTileX         = tree['ElecEvent/fElecChannels/fElecChannels.fxTile'].array(entry_start=0, entry_stop=10)
            self.fTileY         = tree['ElecEvent/fElecChannels/fElecChannels.fyTile'].array(entry_start=0, entry_stop=10)

    def _get_one_event(self, evtid):
        if evtid < self.start_evt or evtid >= self.start_evt + self.load_nentries:
            print("Error: access event out of range!!!")
            return False
        else:
            one_event = {}
            one_event['wf'] = self.fWFAndNoise[evtid - self.start_evt]
            one_event['xpos'] = self.fXPosition[evtid - self.start_evt]
            one_event['ypos'] = self.fYPosition[evtid - self.start_evt]
            one_event['xtile'] = self.fTileX[evtid - self.start_evt]
            one_event['ytile'] = self.fTileY[evtid - self.start_evt]
            one_event['q'] = self.fCharge[evtid - self.start_evt]
            one_event['localid'] = self.fLocalID[evtid - self.start_evt]
            return one_event
        
    
    def _plot_topo2d(self, evtid, noiseFlag=False, numPads=16, padSize=6.0):
        x_clr = 'blue'
        y_clr = 'orange'
        one_event = self._get_one_event(evtid)
        wfs = one_event['wf']
        xpos = one_event['xpos']
        ypos = one_event['ypos']
        charge = one_event['q']
        localids = one_event['localid']
        tilex = one_event['xtile']
        tiley = one_event['ytile']
        for wf, x, y, q, lid in zip(wfs, xpos, ypos, charge, localids):
            if 0 < q < 600 and not noiseFlag:
                continue
            if lid < 16: # It is a x-strip, aligned along with the y-axis
                rectangles = []
                ystart = -(numPads*padSize)/2.
                for i in range(numPads):
                    if q > 600:
                        rectangles.append( plt.Rectangle((x, y+ystart+padSize*i), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=x_clr,ec=(0.,0.,0.,1.),angle=45.) )
                    else:
                        rectangles.append( plt.Rectangle((x, y+ystart+padSize*i), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=x_clr,ec=(0.,0.,0.,1.),angle=45., alpha=0.5) )
                    plt.gca().add_patch(rectangles[i])

            else:
                rectangles = []
                xstart = -(numPads*padSize)/2.
                for i in range(numPads):
                    if q > 600:
                        rectangles.append( plt.Rectangle((x+xstart+padSize*i+padSize/2., y-padSize/2.), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=y_clr,ec=(0.,0.,0.,1.),angle=45.) )
                    else:
                        rectangles.append( plt.Rectangle((x+xstart+padSize*i+padSize/2., y-padSize/2.), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=y_clr,ec=(0.,0.,0.,1.),angle=45., alpha=0.5) )
                    plt.gca().add_patch(rectangles[i])

        txmin, txmax, tymin, tymax = 1e5, -1e5, 1e5, -1e5
        for q, tx, ty in zip(charge, tilex, tiley):
            if 0 < q < 600 and not noiseFlag:
                continue
            if tx > txmax:
                txmax = tx
            if tx < txmin:
                txmin = tx
            if ty > tymax:
                tymax = ty
            if ty < tymin:
                tymin = ty
                
            rec = plt.Rectangle((tx-96/2., ty-96/2.), 96., 96., fc='none', ec='black', lw=0.8)
            plt.gca().add_patch(rec)
            
        plt.xlim(txmin-96, txmax+96)
        plt.ylim(tymin-96, tymax+96)