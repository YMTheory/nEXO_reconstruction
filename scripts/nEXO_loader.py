import uproot as up
import numpy as np

import matplotlib.pyplot as plt

class loader():
    def __init__(self) -> None:
        self.filename = '/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/Baseline2019_bb0n_FullLXe_seed52_comsol.nEXOevents.root'
        self.load_nentries = 50
        self.start_evt = 0

        self.fWFAndNoise    = None
        self.fTileX         = None
        self.fTileY         = None
        self.fLocalID       = None
        self.fXPosition     = None
        self.fYPosition     = None
        self.fCharge        = None
        self.fChannelNTE    = None
        self.fNoiseTag      = None

        self.fEventXpos     = None
        self.fEventYpos     = None
        self.fEventZpos     = None
        self.fDepositE      = None
        
        self.fNESTHitX      = None
        self.fNESTHitY      = None
        self.fNESTHitZ      = None
        self.fNESTHitNTE    = None
        self.fNESTHitType   = None
        
        self.dx             = 0.
        self.dy             = 0.

        # cut the time window for fitting:
        self.fit_t1         = -60
        self.fit_t2         = -10
        
    def _load_event(self):
        with up.open(self.filename) as f:
            tree = f['Event/Elec/ElecEvent']
            #print(tree.keys())
            self.fWFAndNoise    = tree['ElecEvent/fElecChannels/fElecChannels.fWFAndNoise'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fTrueWF        = tree['ElecEvent/fElecChannels/fElecChannels.fTrueWF'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fXPosition     = tree['ElecEvent/fElecChannels/fElecChannels.fXPosition'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fYPosition     = tree['ElecEvent/fElecChannels/fElecChannels.fYPosition'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fCharge        = tree['ElecEvent/fElecChannels/fElecChannels.fChannelCharge'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fLocalID       = tree['ElecEvent/fElecChannels/fElecChannels.fChannelLocalId'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fTileX         = tree['ElecEvent/fElecChannels/fElecChannels.fxTile'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fTileY         = tree['ElecEvent/fElecChannels/fElecChannels.fyTile'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fNoiseTag      = tree['ElecEvent/fElecChannels/fElecChannels.fChannelNoiseTag'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fChannelNTE    = tree['ElecEvent/fElecChannels/fElecChannels.fChannelNTE'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            
            tree = f['Event/Sim/SimEvent']
            self.fEventXpos     = tree['SimEvent/fXpos'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fEventYpos     = tree['SimEvent/fYpos'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fEventZpos     = tree['SimEvent/fZpos'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fDepositE      = tree['SimEvent/fEnergyDeposit'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)

            self.fNESTHitX      = tree['SimEvent/fNESTHitX'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fNESTHitY      = tree['SimEvent/fNESTHitY'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)
            self.fNESTHitZ      = tree['SimEvent/fNESTHitZ'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)    
            self.fNESTHitNTE    = tree['SimEvent/fNESTHitNTE'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)  
            self.fNESTHitType   = tree['SimEvent/fNESTHitType'].array(entry_start=self.start_evt, entry_stop=self.start_evt+self.load_nentries)  

    def _get_one_event(self, evtid):
        if evtid < self.start_evt or evtid >= self.start_evt + self.load_nentries:
            print("Error: access event out of range!!!")
            return False
        else:
            one_event = {}
            one_event['wf'] = self.fWFAndNoise[evtid - self.start_evt]
            one_event['true_wf'] = self.fTrueWF[evtid - self.start_evt]
            one_event['xpos'] = self.fXPosition[evtid - self.start_evt]
            one_event['ypos'] = self.fYPosition[evtid - self.start_evt]
            one_event['xtile'] = self.fTileX[evtid - self.start_evt]
            one_event['ytile'] = self.fTileY[evtid - self.start_evt]
            one_event['q'] = self.fCharge[evtid - self.start_evt]
            one_event['localid'] = self.fLocalID[evtid - self.start_evt]
            one_event['noise_tag'] = self.fNoiseTag[evtid - self.start_evt]
            one_event['event_x'] = self.fEventXpos[evtid - self.start_evt]
            one_event['event_y'] = self.fEventYpos[evtid - self.start_evt]
            one_event['event_z'] = self.fEventZpos[evtid - self.start_evt]
            one_event['event_E'] = self.fDepositE[evtid - self.start_evt]
            one_event['nte'] = self.fChannelNTE[evtid - self.start_evt]
            one_event['nest_hitx'] = self.fNESTHitX[evtid - self.start_evt]
            one_event['nest_hity'] = self.fNESTHitY[evtid - self.start_evt]
            one_event['nest_hitz'] = self.fNESTHitZ[evtid - self.start_evt] 
            one_event['nest_hitnte'] = self.fNESTHitNTE[evtid - self.start_evt] 
            one_event['nest_hittype'] = self.fNESTHitType[evtid - self.start_evt] 
            return one_event
        
    
    def _plot_topo2d(self, evtid, noiseFlag=False, numPads=16, padSize=6.0, truthDepPos=False, trueNESTHitPos=False, fitPos=[], truthPCD=False, PCDs=[]):
        x_clr = 'blue'
        y_clr = 'orange'
        one_event = self._get_one_event(evtid)
        xpos = one_event['xpos']
        ypos = one_event['ypos']
        charge = one_event['q']
        localids = one_event['localid']
        tilex = one_event['xtile']
        tiley = one_event['ytile']
        event_x = one_event['event_x']
        event_y = one_event['event_y']
        event_z = one_event['event_z']
        event_E = one_event['event_E']
        nest_x = one_event['nest_hitx']
        nest_y = one_event['nest_hity']
        nest_n = one_event['nest_hitnte']
        for x, y, q, lid in zip(xpos, ypos, charge, localids):
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
            
        #plt.xlim(txmin-96, txmax+96)
        #plt.ylim(tymin-96, tymax+96)
        
        if truthPCD:
            plt.scatter(PCDs[0], PCDs[1], s=PCDs[2], color='green', alpha=0.3)

        if trueNESTHitPos:
            plt.scatter(nest_x, nest_y, s=nest_n/100, color='green', fc='none', ec='green', lw=0.6)

        if truthDepPos:
            #print(event_x[1], event_y[1], event_z[1])
            #plt.scatter(event_x, event_y, s=np.abs(event_z)/10., color='red', fc='none', ec='red', lw=0.5)
            plt.scatter(event_x, event_y, s=event_E*500, color='red', fc='none', ec='red', lw=0.5)

                
        
        if len(fitPos) > 0:
            fit_x, fit_y, fit_q = fitPos['x'], fitPos['y'], fitPos['q']
            plt.scatter(fit_x+self.dx, fit_y+self.dy, s=fit_q*1000, ec='darkgreen', fc='none', lw=1.0)
            
            
            
            
            
        
    def assembling_for_fitter(self, evtid, strip_type='all', threshold=600):
        times, wfs, sxs, sys, ystrips, qs, tags = [], [], [], [], [], [], []
        event = self._get_one_event(evtid)
        
        e_per_adctick = int(40000./4096.)
        amp = threshold / e_per_adctick
        
        # Set the first collection strip locating as (0, 0)
        dx, dy = 1e6, 1e6
        for q, x, y, tag in zip(event['q'], event['xpos'], event['ypos'], event['noise_tag']):
            if q > 600 and not tag:
                dx, dy = x, y
                self.dx = dx
                self.dy = dy
                break
        if dx == 1e6 and dy == 1e6:
            print("Error: there is no collection strip with charge > 600e- in this event! CHECK REQUIRED!!")
            return None
        
        for q, wf, x, y, lid in zip(event['q'], event['wf'], event['xpos'], event['ypos'], event['localid']):
            if strip_type == 'coll':
                if tag or q < threshold:
                    continue
            elif strip_type == 'indu':
                if q != 0:
                    continue
            elif strip_type == 'all':
                if tag:
                    continue
            else:
                print(f'Error!! Unknown strip type {strip_type}.')
            if lid >= 16:
                ystrips.append(True)
            else:
                ystrips.append(False)
            
            wfs.append(wf)
            sxs.append(x-dx)
            sys.append(y-dy)
            times.append(np.arange(len(wf))*0.5)
            qs.append(q)
            tags.append(tag)
            
        times = np.array(times)
        wfs = np.array(wfs)
        
        
        return times[self.fit_t1:self.fit_t2], wfs[self.fit_t1:self.fit_t2], sxs, sys, ystrips, qs, tags
            
            
    def scanning_points(self, sxs, sys, ystrips, padSize=6):
        # Basic idea is keep the resolution as the size of a pad.
        # The input should be the info of collection strips only.
        # Find all crossing points of x and y strips and extend them a little bit
        
        x_xstrip, y_xstrip, x_ystrip, y_ystrip = [], [], [], []
        for x, y, flag in zip(sxs, sys, ystrips):
            if flag:
                x_ystrip.append(x)
                y_ystrip.append(y)
            else:
                x_xstrip.append(x)
                y_xstrip.append(y)
        print(x_xstrip, y_xstrip)
        print(x_ystrip, y_ystrip)
        x_cross, y_cross = [], []
        for x1, _ in zip(x_xstrip, y_xstrip):
            for _, y2 in zip(x_ystrip, y_ystrip):
                #x_cross.append(x1)
                #y_cross.append(y2)
                x_cross.append(x1-padSize/2.)
                y_cross.append(y2)
                x_cross.append(x1+padSize/2.)
                y_cross.append(y2)
                x_cross.append(x1)
                y_cross.append(y2-padSize/2.)
                x_cross.append(x1)
                y_cross.append(y2+padSize/2.)
          
        x_cross = np.array(x_cross)
        y_cross = np.array(y_cross)      
        return x_cross, y_cross
    

    def scanning_points_fine(self, sxs, sys, ystrips, padSize=6.0):
        x_xstrip, y_xstrip, x_ystrip, y_ystrip = [], [], [], []
        for x, y, flag in zip(sxs, sys, ystrips):
            if flag:
                x_ystrip.append(x)
                y_ystrip.append(y)
            else:
                x_xstrip.append(x)
                y_xstrip.append(y)
        print(x_xstrip, y_xstrip)
        print(x_ystrip, y_ystrip)
        x_cross, y_cross = [], []
        half_step, step = 2, 1.5
        for x1, _ in zip(x_xstrip, y_xstrip):
            for _, y2 in zip(x_ystrip, y_ystrip):
                for i_step in range(int(half_step*2+1)):
                    for j_step in range(int(half_step*2+1)):
                        x_cross.append(x1 - half_step*step + i_step * step)
                        y_cross.append(y2 - half_step*step + j_step * step)

        x_cross = np.array(x_cross)
        y_cross = np.array(y_cross)
        
        return x_cross, y_cross
                    
                    
                    
                    

    def Simple_Zrec(self, evtid, amp_threshold=4000): # both thresholds have units: e-
        one_event = self._get_one_event(evtid)
        wfs = one_event['wf']
        
        e_per_adctick = int(40000./4096.)
        amp_threshold = amp_threshold / e_per_adctick
        
        for ich, wf in enumerate(wfs):
            time = np.arange(0, len(wf), 0.5)
            
            v_drift = 1.70 # unit: mm/us

            for i in range(len(wf)-1):
                pre_wf, post_wf = wf[i], wf[i+1]
                if pre_wf <= amp_threshold and post_wf >= amp_threshold:
                    t = (time[i]+time[i+1]) /2.
                    z = t * v_drift
                    print(f'Passing time is {t} us, the rough drift distance is {z} mm.')
                    return z    
    
        
        