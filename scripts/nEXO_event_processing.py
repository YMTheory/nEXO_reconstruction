# Pre-processing waveforms from nEXO offline simulation
## One point is to group channels based on the scanning points we used in fitting. Only channels are close enough to a the scanning points will be tuned in fitting.

import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import pdist

from scripts.nEXO_loader import loader

#from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
#from sklearn.cluster import OPTICS, cluster_optics_dbscan


class event_builder():
    def __init__(self) -> None:
        self.loader = loader()
        self.mc_event = None
        
        self.collection_strip_group = []
        self.strip_group = []
        
        self.dx = 0.
        self.dy = 0.
        
        # channel level info:
        self.selected_coll_id = None
        self.selected_other_id = None
        self.selected_all_id = None
        
        self.x_cross = None
        self.y_cross = None

        self.time_coll = None
        self.wf_coll = None
        self.strip_x_coll = None
        self.strip_y_coll = None
        self.charge_coll = None
        self.ystrip_flag_coll = None
        self.charge_rec_coll = None
        
        self.time_all = None
        self.wf_all = None
        self.strip_x_all = None
        self.strip_y_all = None
        self.charge_all = None
        self.ystrip_flag_all = None
        self.charge_rec_all = None
        
        self.fit_t1     = 0
        self.fit_t2     = -1
        
        # event level info:
        self.event_q_rec = 0.
        self.event_q_true = 0.

        self.amp_thre = 0.

    def set_filename(self, filename):
        self.loader.filename = filename
        
    def get_mc_event(self, evtid):
        self.loader._load_event()
        self.mc_event = self.loader._get_one_event(evtid)

    def set_load_nentries(self, nentries):
        self.loader.load_nentries = nentries
        
        
    def IsMultiEvent_MCtruth(self, cut=6.0):
        # Use the NEST MC truth to select multi-site events:
        nest_x, nest_y, = self.mc_event['nest_hitx'], self.mc_event['nest_hity']
        points = np.column_stack((nest_x, nest_y))
        distances = pdist(points)
        max_distance = np.max(distances)
        
        if max_distance > cut:
            return True
        else:
            return False

    def set_amp_thre(self, thre):
        self.amp_thre = thre
    

    def channel_selection(self):
        # Some induction channel has very small amplitude which should not be fitted at all..
        # The thre is calculated as ... (require re-calculation later).
        thre = self.amp_thre
        Ncha = len(self.mc_event['wf'])
        self.selected_coll_id, self.selected_other_id, self.selected_all_id = [], [], []
        for i in range(Ncha):
            wf = self.mc_event['wf'][i]
            noisetag = self.mc_event['noise_tag'][i]
            charge = self.mc_event['q'][i]
            if np.any(np.abs(wf) > thre):
                if not noisetag: # Use truth here
                    if charge > 0:
                        self.selected_coll_id.append(i)
                    else:
                        self.selected_other_id.append(i)

        for num in self.selected_coll_id:
            self.selected_all_id.append(num)
        for num in self.selected_other_id:
            self.selected_all_id.append(num)
              
              
              
    def find_intersections(self, x_strip, y_strip, charge, ystrip_flag):
        '''
        Task: given all changes with charges collected, both x-strips and y-strips, find the intersections.
        Input: x-coor, y-coor, charge and if_the_strip_is_y_strip.
        Output: 
        '''
        x_xstrip, y_xstrip, q_xstrip, x_ystrip, y_ystrip, q_ystrip = [], [], [], [], [], []
        for x, y, q, flag in zip(x_strip, y_strip, charge, ystrip_flag) :
            if flag:         
                x_ystrip.append(x)
                y_ystrip.append(y)
                q_ystrip.append(q)
            else:
                x_xstrip.append(x)
                y_xstrip.append(y)
                q_xstrip.append(q)

        #print(f'Total {len(x_xstrip)} x strips and {len(x_ystrip)} y strips.')
        x_cross, y_cross = [], []
        x_cross_all, y_cross_all = [], []
        cross_xstrip_charge_all, cross_ystrip_charge_all = [], []
        cross_xstrip_charge, cross_ystrip_charge = [], []
        strip_length = 96
        for x1, y1, q1 in zip(x_xstrip, y_xstrip , q_xstrip):
            for x2, y2, q2 in zip(x_ystrip, y_ystrip, q_ystrip):
                if np.abs(x1-x2) > strip_length/2. or np.abs(y1-y2)>strip_length/2.:
                    continue
                else:
                    x_cross_all.append(x1)
                    y_cross_all.append(y2)
                    cross_xstrip_charge_all.append(q1)
                    cross_ystrip_charge_all.append(q2)
              
              
        xcharge_dict, ycharge_dict = {}, {}
        for xi, yi, qxi, qyi in zip(x_cross_all, y_cross_all, cross_xstrip_charge_all, cross_ystrip_charge_all):
            if (xi, yi) in xcharge_dict:
                xcharge_dict[(xi, yi)] += qxi
            else:
                xcharge_dict[(xi, yi)] = qxi
            if (xi, yi) in ycharge_dict:
                ycharge_dict[(xi, yi)] += qyi   
            else:
                ycharge_dict[(xi, yi)] = qyi      
              
        keys_are_same = set(xcharge_dict.keys()) == set(ycharge_dict.keys())
        if not keys_are_same:
            print('Keys are not the same!')
        return xcharge_dict, ycharge_dict
      
    
    def group_channels(self, noise=True):
        
        self.channel_selection() # only non-noise channel over threshold remained.

        q_coll_selected = self.mc_event['q'][self.selected_coll_id]
        strip_x_coll_selected = self.mc_event['xpos'][self.selected_coll_id]
        strip_y_coll_selected = self.mc_event['ypos'][self.selected_coll_id]
        locid_coll_selected = self.mc_event['localid'][self.selected_coll_id]
        ystrip_flag_coll_selected = []
        for lid in locid_coll_selected:
            if lid < 16:
                ystrip_flag_coll_selected.append(False)
            else:
                ystrip_flag_coll_selected.append(True)
        ystrip_flag_coll_selected = np.array(ystrip_flag_coll_selected)
        
        # Set the very 1st selected channel center at (0, 0)
        self.dx, self.dy = strip_x_coll_selected[0], strip_y_coll_selected[0]
        strip_x_coll_selected = strip_x_coll_selected - self.dx
        strip_y_coll_selected = strip_y_coll_selected - self.dy
        
        #self.x_cross, self.y_cross, _, _ = self.find_intersections(strip_x_coll_selected, strip_y_coll_selected, q_coll_selected, ystrip_flag_coll_selected)
        #print(f'There are {len(strip_x_coll_selected)} collection strips.')
        #print(f'Total intersections of x and y collection strips is {len(self.x_cross)}.')

        self.time_coll, self.wf_coll, self.strip_x_coll, self.strip_y_coll, self.ystrip_flag_coll, self.charge_coll, = [], [], [], [], [], []
        self.time_all,  self.wf_all,  self.strip_x_all,  self.strip_y_all,  self.ystrip_flag_all,  self.charge_all,  = [], [], [], [], [], []
        
        for q, wf, true_wf, x, y, lid in zip(self.mc_event['q'][self.selected_coll_id], \
                                    self.mc_event['wf'][self.selected_coll_id], \
                                    self.mc_event['true_wf'][self.selected_coll_id], \
                                    self.mc_event['xpos'][self.selected_coll_id], \
                                    self.mc_event['ypos'][self.selected_coll_id], \
                                    self.mc_event['localid'][self.selected_coll_id]\
                                   ):
            self.time_coll.append(np.arange(len(wf))*0.5)
            if noise:
                self.wf_coll.append(wf)
            else:
                self.wf_coll.append(true_wf)
            self.strip_x_coll.append(x-self.dx)
            self.strip_y_coll.append(y-self.dy)
            self.charge_coll.append(q)
            if lid < 16:
                self.ystrip_flag_coll.append(False)
            else:
                self.ystrip_flag_coll.append(True)
        self.time_coll          = np.array(self.time_coll)
        self.wf_coll            = np.array(self.wf_coll)
        self.time_coll          = self.time_coll[:, self.fit_t1:self.fit_t2]
        self.wf_coll            = self.wf_coll[:, self.fit_t1:self.fit_t2]
        self.strip_x_coll       = np.array(self.strip_x_coll)
        self.strip_y_coll       = np.array(self.strip_y_coll)
        self.charge_coll        = np.array(self.charge_coll)
        self.ystrip_flag_coll   = np.array(self.ystrip_flag_coll)
            
        for q, wf, true_wf, x, y, lid in zip(self.mc_event['q'][self.selected_all_id], \
                                    self.mc_event['wf'][self.selected_all_id], \
                                    self.mc_event['true_wf'][self.selected_all_id], \
                                    self.mc_event['xpos'][self.selected_all_id], \
                                    self.mc_event['ypos'][self.selected_all_id], \
                                    self.mc_event['localid'][self.selected_all_id]\
                                   ):
            self.time_all.append(np.arange(len(wf))*0.5)
            if noise:
                self.wf_all.append(wf)
            else:
                self.wf_all.append(true_wf)
            self.strip_x_all.append(x-self.dx)
            self.strip_y_all.append(y-self.dy)
            self.charge_all.append(q)
            if lid < 16:
                self.ystrip_flag_all.append(False)
            else:
                self.ystrip_flag_all.append(True)

        self.time_all          = np.array(self.time_all)
        self.wf_all            = np.array(self.wf_all)
        self.time_all          = self.time_all[:, self.fit_t1:self.fit_t2]
        self.wf_all            = self.wf_all[:, self.fit_t1:self.fit_t2]
        self.strip_x_all       = np.array(self.strip_x_all)
        self.strip_y_all       = np.array(self.strip_y_all)
        self.charge_all        = np.array(self.charge_all)
        self.ystrip_flag_all   = np.array(self.ystrip_flag_all)


        
    def largest_distance_among_crosspoints(self):
        max_distance = 0.
        max_distance_x, max_distance_y = 0, 0
        for xi, yi in zip(self.x_cross, self.y_cross):
            for xj, yj in zip(self.x_cross, self.y_cross):
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d > max_distance:
                    max_distance = d
                    max_distance_x = np.abs(xi - xj)
                    max_distance_y = np.abs(yi - yj)
        print(f'Largest distance of cross points are {max_distance:.3f} mm with dx = {max_distance_x} mm ({int(max_distance_x/6)} strips), dy = {max_distance_y} mm ({int(max_distance_y/6)} strips).')
        
        

    def deposition_clustering(self, method='Kmeans', draw=False, num=-10):
        # Try the ideal case, where the deposition positions are known from MC truth.
        depox, depoy, depoE = self.mc_event['event_x'], self.mc_event['event_y'], self.mc_event['event_E']
        sum_E = np.sum(self.mc_event['event_E'])
        depos, depoE_filtered = [], [] 
        for x, y, E in zip(depox, depoy, depoE):
            if E < 0.005 * sum_E:
                depos.append([x, y])
                depoE_filtered.append(E)
        depos = np.array(depos)
        depoE_filtered = np.array(depoE_filtered)
        
        if method == 'Kmeans':
            inertias = []
            for k in range(1, 21):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(depos)
                inertias.append(kmeans.inertia_)
        
            inertias = np.array(inertias)
            print(inertias)
            best_id = np.argmin(inertias)
            print(f'Best cluster number should be {best_id+1} for this event.')
        
            if num < 1 or num > 20:
                num = best_id + 1
        
            kmeans = KMeans(n_clusters=num, random_state=42)
            kmeans.fit(depos)
        
            labels = kmeans.labels_

        elif method == 'DBSCAN':
            db = DBSCAN(eps=1.0, min_samples=1).fit(depos)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_    = list(labels).count(-1)
            
            num = n_clusters_
            
            #print("Estimated number of clusters: %d" % n_clusters_)
            #print("Estimated number of noise points: %d" % n_noise_)
        
        elif method == 'OPTICS':
            clust = OPTICS(max_eps=1.0)
            clust.fit(depos)
            labels = cluster_optics_dbscan(
                reachability=clust.reachability_,
                core_distances=clust.core_distances_,
                ordering=clust.ordering_,
                eps=1.0,
            )
            unique_elem = np.unique(labels,)
            num = len(unique_elem)
            
            print(f'Estimated number of clusters: {num} from OPTICS.')
            
        
        points_x = [[] for _ in range(num)]
        points_y = [[] for _ in range(num)]
        points_E = [[] for _ in range(num)]
        cluster_x = [0 for _ in range(num)]
        cluster_y = [0 for _ in range(num)]
        cluster_E = [0 for _ in range(num)]
        for i in range(len(depos)):
            for lb in range(num):
                if labels[i] == lb:
                    points_x[lb].append(depos[i, 0])
                    points_y[lb].append(depos[i, 1])
                    points_E[lb].append(depoE_filtered[i])
        for i in range(num):
            cluster_x[i] = np.mean(np.array(points_x[i]))
            cluster_y[i] = np.mean(np.array(points_y[i]))
            cluster_E[i] = np.sum(np.array(points_E[i])) / np.sum(depoE)
        
        if draw:
            _, ax = plt.subplots(figsize=(6, 5))
            colors = ['blue', 'orange', 'green', 'red', 'chocolate', 'crimson']
            for i in range(num):
                ax.scatter(np.array(points_x[i]), np.array(points_y[i]), s=np.array(points_E[i])*5000, fc='none', ec=colors[i], label=f'cluster {i}') 
            ax.set_xlabel('x [mm]', fontsize=13)
            ax.set_ylabel('y [mm]', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(prop={'size':12})
            plt.tight_layout()
        
        return cluster_x-self.dx, cluster_y-self.dy, cluster_E


    def charge_reconstrucion(self, unscaled_wf, electron):
        Z_anode = 403 # mm
        lifetime = 1e4 # us
        velocity = 1.7 # mm/us
        z = -1022.6
        correction = np.exp((np.abs(z)-Z_anode)/velocity/lifetime)
        # In nexo offline, there is a scale facotr 9 on the waveform amplitudes.
        scale = 9.
        #truth = scale * np.asarray(unscaled_truth)
        wf = scale * np.asarray(unscaled_wf)
        gain = 3.03
        #if electron != 0:   
        #    print(f'This channel collected {electron} electrons.')
        #    print(f'But the integral of the true waveform is {np.sum(wf) / gain}.')
        return np.sum(wf) / gain
        
        
    def simple_reconstruction(self):
        event_q = 0.
        channel_rec_q = []
        Ncha = len(self.mc_event['q'])
        for i in range(Ncha):
            channel_q = self.charge_reconstrucion(self.mc_event['true_wf'][i], self.mc_event['nte'][i]) 
            channel_rec_q.append(channel_q)
            event_q += channel_q
            
        self.event_q_rec = event_q
        self.event_q_true = np.sum(self.mc_event['q'])

        channel_rec_q = np.array(channel_rec_q)
        self.mc_event['q_rec'] = channel_rec_q
        
        x_cross, y_cross, cross_xstrip_charge, cross_ystrip_charge = self.find_intersections(self.strip_x_all, self.strip_y_all, self.charge_all, self.ystrip_flag_all)

        return self.event_q_rec, self.event_q_true, x_cross, y_cross, cross_xstrip_charge, cross_ystrip_charge
        

    def _plot_channels2D(self, fitx=[], fity=[], fitq=[], truthDep=False, truthNEST=False):
        x_clr = 'blue'
        y_clr = 'orange'
        xpos        = self.mc_event['xpos'][self.selected_coll_id]
        ypos        = self.mc_event['ypos'][self.selected_coll_id]
        charge      = self.mc_event['q'][self.selected_coll_id]
        localids    = self.mc_event['localid'][self.selected_coll_id]
        tilex       = self.mc_event['xtile'][self.selected_coll_id]
        tiley       = self.mc_event['ytile'][self.selected_coll_id]
        numPads, padSize = 16, 6
        for x, y, q, lid in zip(xpos, ypos, charge, localids):
            #x = x - self.dx
            #y = y - self.dy
            if lid < 16: # It is a x-strip, aligned along with the y-axis
                rectangles = []
                ystart = -(numPads*padSize)/2.
                for i in range(numPads):
                    rectangles.append( plt.Rectangle((x, y+ystart+padSize*i), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=x_clr,ec=(0.,0.,0.,1.),angle=45., alpha=1.0) )
                    plt.gca().add_patch(rectangles[i])
                #plt.plot(x, y, '*', color='brown', markersize=8, alpha=0.5)

            else:
                rectangles = []
                xstart = -(numPads*padSize)/2.
                for i in range(numPads):
                    rectangles.append( plt.Rectangle((x+xstart+padSize*i+padSize/2., y-padSize/2.), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=y_clr,ec=(0.,0.,0.,1.),angle=45., alpha=1.0) )
                    plt.gca().add_patch(rectangles[i])
                #plt.plot(x, y, 'd', color='crimson', markersize=8, alpha=0.5)
        txmin, txmax, tymin, tymax = 1e5, -1e5, 1e5, -1e5
        for tx, ty in zip(tilex, tiley):
            #tx = tx - self.dx
            #ty = ty - self.dy
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
        
        xpos        = self.mc_event['xpos'][self.selected_other_id]
        ypos        = self.mc_event['ypos'][self.selected_other_id]
        charge      = self.mc_event['q'][self.selected_other_id]
        localids    = self.mc_event['localid'][self.selected_other_id]
        tilex       = self.mc_event['xtile'][self.selected_other_id]
        tiley       = self.mc_event['ytile'][self.selected_other_id]

        numPads, padSize = 16, 6
        for x, y, q, lid in zip(xpos, ypos, charge, localids):
            #x = x - self.dx
            #y = y - self.dy
            if lid < 16: # It is a x-strip, aligned along with the y-axis
                rectangles = []
                ystart = -(numPads*padSize)/2.
                for i in range(numPads):
                    rectangles.append( plt.Rectangle((x, y+ystart+padSize*i), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=x_clr,ec=(0.,0.,0.,1.),angle=45., alpha=0.5) )
                    plt.gca().add_patch(rectangles[i])
                #plt.plot(x, y, '*', color='brown', markersize=8, alpha=0.5)

            else:
                rectangles = []
                xstart = -(numPads*padSize)/2.
                for i in range(numPads):
                    rectangles.append( plt.Rectangle((x+xstart+padSize*i+padSize/2., y-padSize/2.), padSize/np.sqrt(2), padSize/np.sqrt(2), fc=y_clr,ec=(0.,0.,0.,1.),angle=45., alpha=0.5) )
                    plt.gca().add_patch(rectangles[i])
                #plt.plot(x, y, 'd', color='crimson', markersize=8, alpha=0.5)

        

        txmin, txmax, tymin, tymax = 1e5, -1e5, 1e5, -1e5
        for tx, ty in zip(tilex, tiley):
            #tx = tx - self.dx
            #ty = ty - self.dy
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
        
        if truthDep:
            event_x     = self.mc_event['event_x']
            event_y     = self.mc_event['event_y']
            event_z     = self.mc_event['event_z']
            event_E     = self.mc_event['event_E']
            sum_E       = np.sum(event_E)
            plt.scatter(event_x-self.dx, event_y-self.dy, s=event_E/sum_E*100, fc='none', ec='green', alpha=0.8, label='true deposition')

        if truthNEST:
            nest_x, nest_y, nest_q = self.mc_event['nest_hitx'], self.mc_event['nest_hity'], self.mc_event['nest_hitnte']
            print(f'Draw the NEST truth hits: total {len(nest_x)} hits.')
            plt.scatter(nest_x, nest_y, s=nest_q/100, fc='none', ec='green', alpha=0.8, label='true NEST')
        
        
        edges = ['red', 'black', 'pink', 'green']
        if len(fitx) != 0:
            for i, (x, y, q) in enumerate(zip(fitx, fity, fitq)):
                plt.scatter(x, y, s=q*100, fc='none', ec=edges[i], lw=2, label=f'{i+1}-cluster')
        
        
        plt.legend(prop={'size':12})
        
        plt.tight_layout()
        

    def _plot_channel_waveforms(self, noise=True, xmin=300, xmax=400):
        n_chanenls = len(self.wf_all)
        nrow = int(n_chanenls/5) + 1
        
        fig, ax = plt.subplots(nrow, 5, figsize=(15, 3*nrow))
        for i in range(n_chanenls):
            irow = int(i/5)
            icol = int(i%5)
            ax[irow, icol].plot(self.time_all[i], self.wf_all[i], label=f'Q={self.charge_all[i]}')
            ax[irow, icol].set_xlabel('time [us]', fontsize=10)
            ax[irow, icol].set_ylabel('current amplitude [ADC]', fontsize=10)
            ax[irow, icol].tick_params(axis='both', labelsize=8)
            ax[irow, icol].legend(fontsize=10)
            type = 'Y' if self.ystrip_flag_all[i] else 'X'
            ax[irow, icol].set_title(f'{type}-strip @({self.strip_x_all[i]+self.dx}, {self.strip_y_all[i]+self.dy})', fontsize=12)
            ax[irow, icol].set_xlim(xmin, xmax)
        plt.tight_layout()
        
        return fig
        
