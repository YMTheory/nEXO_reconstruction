import numpy as np
import time
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares

from scripts.PDF_loader import loader
from scripts.nEXO_event_processing import event_builder


class pcd_fitter():
    
    def __init__(self, charge_x=0.0, charge_y=0.0, charge_z=-622.0, charge_q=1e5):

        self.SamplingInterval               = 0.5 # us

        self.charge_x                       = charge_x  
        self.charge_y                       = charge_y  
        self.charge_z                       = charge_z  
        self.charge_q                       = charge_q  
        
        self.builder    = event_builder()
        self.load       = loader(charge_x=charge_x, charge_y=charge_y, charge_z=charge_z, charge_q=charge_q)

        self.offline_simulation_filename    = None
        self.event_id                       = 0
        self.time_array                     = None
        self.waveform_array                 = None
        self.strip_x_array                  = None
        self.strip_y_array                  = None
        self.strip_type_array               = None 
        self.strip_charge_array             = None  
        self.total_charge_truth             = 0.0
        
        # Fitting configuration parameters       
        self.fit_nchannels                  = 0
        self.fit_channelsId                 = []
        self.fit_tmin                       = 0
        self.fit_tmax                       = 1500

        # Very initial values for fitting ranges:
        self.dt_fixed_flag                   = False
        self.dx_fixed_flag                   = False
        self.dy_fixed_flag                   = False
        self.Q_scale_fixed_flag              = False
        
        self.dt_range_low                    = -10.
        self.dt_range_high                   = 10.
        self.dx_range_low                    = -6.0
        self.dx_range_high                   = 6.0
        self.dy_range_low                    = -6.0
        self.dy_range_high                   = 6.0
        self.Q_scale_range_low               = 0.5
        self.Q_scale_range_high              = 1.5


    # Setters
    def _set_offline_filename(self, filename):
        self.offline_simulation_filename = filename
        
    def _set_event_id(self, event_id):
        self.event_id = event_id
        
    def _set_fit_channels(self, channels_id):
        self.fit_channelsId = channels_id
        self.fit_nchannels = len(channels_id)
        
    
    def _set_fit_time_window(self, tmin, tmax):
        self.fit_tmin = tmin
        self.fit_tmax = tmax
    
    
    def load_one_event(self):
        self.builder.set_filename(self.offline_simulation_filename)
        self.builder.get_mc_event(self.event_id)
        self.builder.group_channels()
        self.time_array = self.builder.time_all[self.fit_channelsId]
        self.waveform_array = self.builder.wf_all[self.fit_channelsId]

        # Set time range for fitting
        min_index, max_index = int(self.fit_tmin/self.SamplingInterval), int(self.fit_tmax/self.SamplingInterval)+1

        self.time_array = self.time_array[:,min_index:max_index]
        self.waveform_array = self.waveform_array[:,min_index:max_index]

        self.strip_x_array = self.builder.strip_x_all[self.fit_channelsId] + self.builder.dx
        self.strip_y_array = self.builder.strip_y_all[self.fit_channelsId] + self.builder.dy
        self.strip_type_array = []
        for elm in self.builder.ystrip_flag_all[self.fit_channelsId]:
            self.strip_type_array.append(not(elm))
        self.strip_type_array = np.array(self.strip_type_array)
        self.strip_charge_array = self.builder.charge_all[self.fit_channelsId]
        self.total_charge_truth = np.sum(self.builder.charge_all)



    def _set_dt_fixed(self, flag):
        self.dt_fixed_flag = flag
        
    def _set_dx_fixed(self, flag):
        self.dx_fixed_flag = flag
        
    def _set_dy_fixed(self, flag):
        self.dy_fixed_flag = flag
        
    def _set_Q_scale_fixed(self, flag):
        self.Q_scale_fixed_flag = flag    


    def _set_dt_range(self, low, high):
        self.dt_range_low = low
        self.dt_range_high = high
        
        
    def _set_dx_range(self, low, high):
        self.dx_range_low = low 
        self.dx_range_high = high
        
    def _set_dy_range(self, low, high):
        self.dy_range_low = low 
        self.dy_range_high = high   
    
    def _set_Q_scale_range(self, low, high):    
        self.Q_scale_range_low = low                    
        self.Q_scale_range_high = high                   
        

    def onePC_fitting(self, t0, x0, y0, Q0):
        
        # For each one PC (point charge), we assign 4 fitting parameters: t0, x0, y0, Q0 (z0 is fixed as the diffusion is correlated with z0)
        
        def create_models(sx, sy, type):
            # Type: is if a x strip?
            def one_model(t, t0, x0, y0, Q0):
                dx, dy = sx - x0, sy - y0
                fit_time = t+t0
                fit_wf = self.load.diffused_waveform_oneChannel(dx, dy, type, t+t0) * Q0
                #return np.interp(t, fit_time, fit_wf)
                return np.interp(t, t, fit_wf)
            return one_model
                
        # Prepare the model list for all channels to be fitter:
        model_list = []
        for i in range(self.fit_nchannels):
            model_list.append(create_models(self.strip_x_array[i], self.strip_y_array[i], not(self.strip_type_array[i])))
        
        least_squares = []
        for icha in range(self.fit_nchannels):
            t, wf = self.time_array[icha], self.waveform_array[icha]
            wferr = np.ones(len(wf)) * (100 / int(40000./4096.)) # Considering the noise level as the errors.
            model = model_list[icha]
            ls = LeastSquares(t, wf, wferr, model)
            least_squares.append(ls)
            
        least_squares_total = least_squares[0]
        for i in range(len(least_squares)-1):
            least_squares_total += least_squares[i+1]
            
        m = Minuit(least_squares_total, t0=t0, x0=x0, y0=y0, Q0=Q0)
        m.print_level = 1

        m.fixed['t0'] = self.dt_fixed_flag
        m.limits['t0'] = (self.dt_range_low, self.dt_range_high)
        m.fixed['x0'] = self.dx_fixed_flag
        m.limits['x0'] = (self.dx_range_low, self.dx_range_high)
        m.fixed['y0'] = self.dy_fixed_flag 
        m.limits['y0'] = (self.dy_range_low, self.dy_range_high)
        m.fixed['Q0'] = self.Q_scale_fixed_flag
        m.limits['Q0'] = (self.Q_scale_range_low, self.Q_scale_range_high)

        m.migrad()
        
        return m


    def onePC_nofitting(self, t0, x0, y0, Q0, draw=False):
        
        # For each one PC (point charge), we assign 4 fitting parameters: t0, x0, y0, Q0 (z0 is fixed as the diffusion is correlated with z0)
        
        def create_models(sx, sy, type):
            # Type: is if a x strip?
            def one_model(t, t0, x0, y0, Q0):
                dx, dy = sx - x0, sy - y0
                fit_time = t+t0
                fit_wf = self.load.diffused_waveform_oneChannel(dx, dy, type, fit_time) * Q0
                #return np.interp(t, fit_time, fit_wf)
                return np.interp(t, t, fit_wf)
            return one_model
                
        # Prepare the model list for all channels to be fitter:
        model_list = []
        for i in range(self.fit_nchannels):
            model_list.append(create_models(self.strip_x_array[i], self.strip_y_array[i], not(self.strip_type_array[i])))
        
        least_squares = []
        generated_wfs = []
        for icha in range(self.fit_nchannels):
            t, wf = self.time_array[icha], self.waveform_array[icha]
            wferr = np.ones(len(wf)) * (100 / int(40000./4096.)) # Considering the noise level as the errors.
            model = model_list[icha]
            one_gen_wf = model(t, t0, x0, y0, Q0)
            generated_wfs.append(one_gen_wf)


        if draw:
            ncol = 4
            nrow = int((self.fit_nchannels-1)/ncol) + 1
            fig, ax = plt.subplots(nrow, ncol, figsize=(16, 5*nrow))

            for ich in range(self.fit_nchannels):
                icol = ich % ncol
                irow = int(ich / ncol)
                if nrow == 1:
                    ax0 = ax[icol]
                else:
                    ax0 = ax[irow, icol]

                ax0.plot(self.time_array[ich], self.waveform_array[ich], ":", label='Offline')
                ax0.plot(t, generated_wfs[ich], label='Fitted')
                ax0.set_xlabel('Time [us]', fontsize=10)
                ax0.set_ylabel('ADC', fontsize=10)
                if self.strip_type_array[ich]:
                    ty = 'xstrip'
                else:
                    ty = 'ystrip'
                ax0.set_title(f'{ty}@ ({self.strip_x_array[ich]:.1f}, {self.strip_y_array[ich]:.1f})', fontsize=11)
                ax0.legend(fontsize=11)
                ax0.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout()
            
            return fig 
        else:
            return 0.    
            