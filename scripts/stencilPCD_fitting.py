import numpy as np
import time
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares

from scripts.PDF_loader import loader
from scripts.nEXO_event_processing import event_builder


class pcd_fitter():
    
    def __init__(self):

        self.SamplingInterval               = 0.5 # us

        self.builder    = event_builder()
        self.load       = loader()

        self.offline_simulation_filename    = None
        self.event_id                       = 0
        self.time_array                     = None
        self.waveform_array                 = None
        self.strip_x_array                  = None
        self.strip_y_array                  = None
        self.strip_type_array               = None  # Array later, True for x-strip, False for y-strip.
        self.strip_charge_array             = None  
        self.total_charge_truth             = 0.0
        
        # Fitting configuration parameters       
        self.fit_nchannels                  = 0
        self.fit_channelsId                 = []
        self.fit_tmin                       = 0
        self.fit_tmax                       = 1500
        self.waveform_noise                 = True
        self.fit_inductive                  = True
        self.fit_pdf_fine                   = True
        self.amplitude_threshold            = 0

        # Very initial values for fitting ranges:
        self.dt_fixed_flag                  = False
        self.dx_fixed_flag                  = False
        self.dy_fixed_flag                  = False
        self.Q_scale_fixed_flag             = False
        
        self.dt_range_low                   = -10.
        self.dt_range_high                  = 10.
        self.dx_range_low                   = -6.0
        self.dx_range_high                  = 6.0
        self.dy_range_low                   = -6.0
        self.dy_range_high                  = 6.0
        self.Q_scale_range_low              = 0.5
        self.Q_scale_range_high             = 1.5
        self.SS_cut                         = 6.0
        
        self.m_fit                          = None  
        self.m_nfitdata                     = 0
        self.m_nparam                       = 0
        self.m_chi2NDF                      = 0.0
        
        self.verbose                        = False

    def clean(self):
        self.time_array                     = None
        self.waveform_array                 = None
        self.strip_x_array                  = None
        self.strip_y_array                  = None
        self.strip_type_array               = None  # Array later, True for x-strip, False for y-strip.
        self.strip_charge_array             = None  
        self.total_charge_truth             = 0.0
        
        # Fitting configuration parameters       
        self.fit_nchannels                  = 0
        self.fit_channelsId                 = []


    # Setters
    def _set_offline_filename(self, filename):
        self.offline_simulation_filename = filename
        self.builder.set_filename(filename)
        
    def _set_event_id(self, event_id):
        self.event_id = event_id
        
    def _set_loading_nevents(self, nevents):
        self.builder.set_load_nentries(nevents)
        
    def _set_fit_channels(self, channels_id):
        self.fit_channelsId = channels_id
        self.fit_nchannels = len(channels_id)
        
    def _set_noise_flag(self, flag):
        self.waveform_noise = flag

    def _set_pdf_fine(self, fine):
        self.fit_pdf_fine = fine
        self.load.PDF_fine = fine
    
    def _set_fit_time_window(self, tmin, tmax):
        self.fit_tmin = tmin
        self.fit_tmax = tmax

    def _set_fit_inductive_flag(self, flag):
        self.fit_inductive = flag

    def _set_SS_cut(self, cut):
        self.SS_cut = cut
    
    def _IsMultiSite(self):
        self.builder.get_mc_event(self.event_id)
        if self.builder.IsMultiEvent_MCtruth(self.SS_cut):
            return True
        else:
            return False

    def _set_amplitude_threshold(self, thre):
        self.amplitude_threshold = thre
        self.builder.set_amp_thre(thre)

    def _set_verbose(self, flag):
        self.verbose = flag
    
    def load_one_event(self):
        # Build event:
        self.builder.get_mc_event(self.event_id)
        self.builder.group_channels(noise=self.waveform_noise)

        if self.fit_inductive:
            n_build_channel = len(self.builder.selected_all_id)
            if self.verbose:
                print(f'===> All channel fitting: total {n_build_channel} channels.')

            # Test channel id:
            for chaid in self.fit_channelsId:
                if chaid > n_build_channel:
                    print(f"Error: channel id {chaid} is out of range from event builder!")
        
            if len(self.fit_channelsId) == 0:
                # If no channel is specified, use all channels in the event.
                self.fit_channelsId = range(n_build_channel)
                self.fit_nchannels = n_build_channel
        
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

        else:
            n_build_channel = len(self.builder.selected_coll_id)
            if self.verbose:
                print(f'===> Collection-only channel fitting: total {n_build_channel} channels.')

            # Test channel id:
            for chaid in self.fit_channelsId:
                if chaid > n_build_channel:
                    print(f"Error: channel id {chaid} is out of range from event builder!")
        
            if len(self.fit_channelsId) == 0:
                # If no channel is specified, use all channels in the event.
                self.fit_channelsId = range(n_build_channel)
                self.fit_nchannels = n_build_channel
        
            self.time_array = self.builder.time_coll[self.fit_channelsId]
            self.waveform_array = self.builder.wf_coll[self.fit_channelsId]

            # Set time range for fitting
            min_index, max_index = int(self.fit_tmin/self.SamplingInterval), int(self.fit_tmax/self.SamplingInterval)+1

            self.time_array = self.time_array[:,min_index:max_index]
            self.waveform_array = self.waveform_array[:,min_index:max_index]

            self.strip_x_array = self.builder.strip_x_coll[self.fit_channelsId] + self.builder.dx
            self.strip_y_array = self.builder.strip_y_coll[self.fit_channelsId] + self.builder.dy
            self.strip_type_array = []
            for elm in self.builder.ystrip_flag_coll[self.fit_channelsId]:
                self.strip_type_array.append(not(elm))
            self.strip_type_array = np.array(self.strip_type_array)
            self.strip_charge_array = self.builder.charge_coll[self.fit_channelsId]
            self.total_charge_truth = np.sum(self.builder.charge_coll)

        # Set the fitting data number:
        self.m_nfitdata = np.sum(np.array([len(wf) for wf in self.waveform_array]))
        
        if self.verbose:
            print(f'===> Total fit data number is {self.m_nfitdata}.')


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

    ## Getter
    def _get_fit_status(self):
        return self.m_fit.valid
        
    def _get_chi2NDF(self):
        # Calculate the reduced chi2 manually:
        self.m_chi2NDF = self.m_fit.fval / ( self.m_nfitdata - self.m_nparam) 
        return self.m_chi2NDF

    def _get_filename(self):
        return self.filename

    def _get_evtid(self):
        return self.event_id

    def onePC_fitting(self, t0, x0, y0, Q0):
        
        # For each one PC (point charge), we assign 4 fitting parameters: t0, x0, y0, Q0 (z0 is fixed as the diffusion is correlated with z0)
        
        def create_models(sx, sy, type):
            # Type: true for x-strip, false for y-strip.
            def one_model(t, t0, x0, y0, Q0): # x0, y0, Q0 are the point charge info.
                dx, dy = sx - x0, sy - y0
                if not type:
                    dx, dy = sy-y0, sx-x0
                fit_time = t+t0
                fit_wf = self.load.diffused_waveform_oneChannel(dx, dy, t) * Q0
                return np.interp(t, fit_time, fit_wf)
            return one_model
                
        # Prepare the model list for all channels to be fitter:
        model_list = []
        for i in range(self.fit_nchannels):
            model_list.append(create_models(self.strip_x_array[i], self.strip_y_array[i], self.strip_type_array[i]))
        
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

        self.m_nparam = 4
        if self.dt_fixed_flag:
            self.m_nparam -= 1
        if self.dx_fixed_flag:
            self.m_nparam -= 1
        if self.dy_fixed_flag:
            self.m_nparam -= 1
        if self.Q_scale_fixed_flag:
            self.m_nparam -= 1
        
        if self.verbose:
            print(f"===> Total fitting paramters number is {self.m_nparam}.")
            
        self.m_fit = m
        return m


    def onePC_nofitting(self, t0, x0, y0, Q0, draw=False):
        
        # For each one PC (point charge), we assign 4 fitting parameters: t0, x0, y0, Q0 (z0 is fixed as the diffusion is correlated with z0)
        
        def create_models(sx, sy, type):
            # Type: true for x-strip, false for y-strip.
            def one_model(t, t0, x0, y0, Q0): # x0, y0, Q0 are the point charge info.
                dx, dy = sx - x0, sy - y0
                if not type:
                    dx, dy = sy-y0, sx-x0
                fit_time = t+t0
                fit_wf = self.load.diffused_waveform_oneChannel(dx, dy, t) * Q0
                return np.interp(t, fit_time, fit_wf)
            return one_model
                
        # Prepare the model list for all channels to be fitter:
        model_list = []
        for i in range(self.fit_nchannels):
            model_list.append(create_models(self.strip_x_array[i], self.strip_y_array[i], self.strip_type_array[i]))
        
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
            
