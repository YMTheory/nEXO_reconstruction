import numpy as np
import uproot as up
import os
import glob
import argparse
import scipy as sc

class event_data:

  __slots__ = ['file', 'number', 'waveform', 'signal', 'event_electrons', 'channel_electrons', 'coordinates', 'noise_tag', 'E', 'lineage']
  
  def __init__(self, file, number) -> None:
    self.file = file
    self.number = number
    
  # N.B: due to a recent uproot update, the WVFM and signal are read out as STLVectors, and so can't be multiplied by 9 yet - this happens in the channel loop of 'event_analysis'
  def get_waveform(self):
    root =  up.open(self.file)
    tree = root['Event/Elec/ElecEvent']
    branch = tree['ElecEvent/fElecChannels']
    self.waveform = branch['fElecChannels.fWFAndNoise'].array(library='np', entry_start=self.number,entry_stop=self.number+1)[0]

  def get_channel_electrons(self):
    root = up.open(self.file)
    self.channel_electrons = root['Event/Elec/ElecEvent/ElecEvent/fElecChannels/fElecChannels.fChannelNTE'].array(library='np', entry_start=self.number, entry_stop=self.number+1)[0]

  def get_event_electrons(self):
    root = up.open(self.file)
    self.event_electrons = (root['Event/Sim/SimEvent/fNTE']).array(library='np', entry_start=self.number, entry_stop=self.number+1)

  # N.B: due to a recent uproot update, the WVFM and signal are read out as STLVectors, and so can't be multiplied by 9 yet - this happens in the channel loop of 'event_analysis'
  def get_signal(self):
    root =  up.open(self.file)
    tree = root['Event/Elec/ElecEvent']
    branch = tree['ElecEvent/fElecChannels']
    self.signal = branch['fElecChannels.fTrueWF'].array(library='np', entry_start=self.number,entry_stop=self.number+1)[0]

  def get_noise_tag(self):
    root = up.open(self.file)
    self.noise_tag = root['Event/Elec/ElecEvent/ElecEvent/fElecChannels/fElecChannels.fChannelNoiseTag'].array(library='np', entry_start=self.number, entry_stop=self.number+1)[0]

  def get_coordinates(self):
    root = up.open(self.file)
    self.coordinates = [np.asarray(root['Event/Sim/SimEvent/SimEvent/fGenX'])[self.number], np.asarray(root['Event/Sim/SimEvent/SimEvent/fGenY'])[self.number], np.asarray(root['Event/Sim/SimEvent/SimEvent/fGenZ'])[self.number]]

  def get_E(self):
    root = up.open(self.file)
    self.E = root['Event/Sim/SimEvent/fTotalEventEnergy'].array(library='np', entry_start=self.number, entry_stop=self.number+1)[0]

  def get_lineage(self):
    root = up.open(self.file)
    self.lineage = root['Event/Sim/SimEvent/fNESTLineageType'].array(library='np', entry_start=self.number, entry_stop=self.number+1)[0]


# Finds amplitude of pulse using omptimum filtering technique according to Golwala's thesis
def optimal_filter(v, s, J, t0, Fs = 2e6): 
    
    '''
    This function takes a data surface (v) index as [channel, time], an expected signal stencil (s) indexed the same, a noise profile (J)
    and a time shift of the expected signal (t0) and then returns the amplitude of the signal in the data set
    using the optimum filter algorithm in Golwala's thesis (https://www.slac.stanford.edu/exp/cdms/ScienceResults/Theses/golwala.pdf) - Apendix B
    '''

    N = len(v[0,:])
    T = N/Fs
    freq = np.fft.rfftfreq(N,1/Fs) 
    
    norm = np.sqrt(2/N) # Using factor of 2 to account for rfft
    
    # Assuming that noise is constant across space, we skip the spatial FFT
    s_fft = np.fft.rfft(s, axis=1)*norm
    v_fft = np.fft.rfft(v, axis=1)*norm
    
    num=0
    denom=0

    for chan in range(len(v[:,0])):
      for i, f in enumerate(freq): 

        if i == 0: continue # Skipping DC offset
      
        num += np.exp(2j*np.pi*f*t0)*np.conjugate(s_fft[chan, i])*v_fft[chan, i]/J[i]
        denom += np.abs(s_fft[chan, i])**2/J[i]
        
    A = num/denom
    
    return A


def get_noise(noise_count):
  
  '''This function opens up the correct noise library and pulls out noise_count many traces from it.'''

  file = up.open("../nexo_offline_waveform_study/expected_1_2_us_noise_lib_100e.root")
  noise = np.asarray(file['noiselib']['noise_int'])[0:noise_count]
  del file # Deleting the file after we save our traces to keep memory down
  return noise  


def add_noise(truth, scale=1):
  length = truth.size
  full_noise = get_noise(10000) # Choosing high noise count to ensure randomness
  flat_noise = np.ndarray.flatten(full_noise)
  start_point = np.random.randint(0,10000-length)
  pre_noise = flat_noise[start_point:start_point+length]
  np.random.shuffle(pre_noise)
  noise_trace = pre_noise*scale
  return truth+noise_trace


def event_analysis(std_truth, J, Z, noise_tag, electrons, cut):

  '''
  This function applies an optimal filter analysis to an event on the channel level. It first checks if the channel is noise, or has signal below the
  given threshold. Then applies the filter to the channel to reconstruct the collected charge and subtrcts from it the true collected charge.
  '''

  scale = 9/3.03
  RMS = 1.
  Z_anode = 403 #mm
  lifetime = 1e4 #us
  velocity = 1.71 #mm/us
  correction  = np.exp((np.abs(Z)-Z_anode)/(lifetime*velocity)) 

  collection = electrons*(1-noise_tag)
  collection_index = np.where(noise_tag==0)[0]

  true_surface = np.zeros((collection.size,len(std_truth[0]))) # Detector's true current surface indexed as [channel, time sample]
  surface = np.zeros((collection.size,len(std_truth[0]))) # Detector's noisey current surface with indexed as [channel, time sample]

  for i, cindex in enumerate(collection_index):
    
    if len(std_truth[cindex]) == 0:
      true_surface[i] = np.zeros(len(std_truth[0])) # If we hit the empty WVFM bug, the zeros here will die in the convolution of the opt filter
      surface[i] = np.zeros(len(std_truth[0]))
      continue
    
    true_surface[i,:] = scale*np.asarray(std_truth[cindex]) # Assigned the non-noise channels to the detector surface
    surface[i,:] = add_noise(true_surface[i,:], scale = RMS) # Adding noise to noisy surface
  
  surface_stencil = true_surface/np.max(true_surface) # Zeros from empty channel glitch will remain zeros here and die as planned

  A = optimal_filter(surface, surface_stencil, J, 0)
  recon_amp = np.real(A)

  print('Reconstructed amplitude is {0}'.format(recon_amp))
  print('True amplitude of the signal is {0}'.format(np.max(true_surface)))

  event_reconstruction = recon_amp*np.sum(surface_stencil) - np.sum(true_surface)
  event_reconstruction *= correction

  return event_reconstruction


def save_data(reconstruction, job_num):
  current_run_file  = len(os.popen('ls -d ./jobs/*/').read().split('\n'))-1
  file_name = './data/Run_'+str(current_run_file)+'/nEXO_Charge_analysis_file_' +str(job_num) + '.npy'
  np.save(file_name, reconstruction, allow_pickle=True)


def start_job(job_num, cut):

  file_list = glob.glob('../strip_lib_retry/*.root')
  file = file_list[job_num]
  print('Analyizing file number ', job_num)
  print('\n')
  print("File name: {}".format(file))
  print('\n')

  # Loading event data for fiducial and signal checks
  temp = up.open(file)
  temp_noise_tag = temp['Event/Elec/ElecEvent/ElecEvent/fElecChannels/fElecChannels.fChannelNoiseTag'].array(library='np')
  event_count = temp_noise_tag.shape[0]
  del temp
  del temp_noise_tag

  reconstruction = np.zeros(event_count)

  for e in range(event_count):

    data = event_data(file, e)

    print('\n')
    print('*'*20)
    print('*'*20)
    print('\n')
    print('Beginning event number: ', e)
    print('\n')
    print('*'*20)
    print('*'*20)
    print('\n')

    data.get_signal()
    truth = data.signal

    data.get_coordinates()
    X = data.coordinates[0]    
    Y = data.coordinates[1]    
    Z = data.coordinates[2]    
    
    data.get_noise_tag()
    noise_tag = data.noise_tag
    
    data.get_channel_electrons()
    electrons = data.channel_electrons

    # Following values taken from https://github.com/nEXO-collaboration/nexo-offline/issues/135
    Z_anode = 403 # bottom of the anode in mm
    Z_cathode = 1586 # top of the cathode in mm
    R_rings = 556 # Inner edge of field shaping rings in mm
    fiducial_cut = 20 # Length of fiducial cut in mm

    # Fiducial cuts
    if np.sqrt(X**2+Y**2) > R_rings-fiducial_cut or abs(Z) < Z_anode+fiducial_cut or abs(Z) > Z_cathode-fiducial_cut:
      print('Event outside of fiducial volume')
      reconstruction[e] = np.nan
      continue
    
    # Pull noise from library to generate PSD
    N = len(truth[0])
    noise_traces = get_noise(10000)[0:N,:]
    noise_FFT = np.abs(np.fft.rfft(noise_traces, axis = 1))**2
    Fs = 2e6
    norm = 2/(N*Fs)
    J = np.mean(noise_FFT, axis = 0)*norm
    
    reconstruction[e] = event_analysis(truth, J, Z, noise_tag, electrons, cut)

    print('\n')
    print('-'*20)
    print('Reconstruction error for this event: ', reconstruction[e])
    print('-'*20)
    print('\n')

    # Removing nan values may be due to fiducial cuts or incomplete waveforms and saving data
    save_data(reconstruction[~np.isnan(reconstruction)], job_num)


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description=__doc__)
  parser.add_argument("jobnum", type=int, help='Job number to be run')
  parser.add_argument("cut", type=int, help='The threshold value to use in the analysis (units of e^-)')
  args = parser.parse_args()

  start_job(args.jobnum, args.cut)

if __name__ == "__main__":
  main()
