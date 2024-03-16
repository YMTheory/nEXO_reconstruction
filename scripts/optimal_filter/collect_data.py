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
    This function takes a data set (v), an expected signal (s), a noise profile (J)
    and a time shift of the expected signal (t0) and then returns the amplitude of the signal in the data set
    using the optimum filter algorithm in Golwala's thesis (https://www.slac.stanford.edu/exp/cdms/ScienceResults/Theses/golwala.pdf) - Apendix B
    '''
    N = len(v)
    T = N/Fs
    freq = np.fft.rfftfreq(N,1/Fs) 
    
    norm = np.sqrt(2/N) # Using factor of 2 to account for rfft
    s_fft = np.fft.rfft(s)*norm
    v_fft = np.fft.rfft(v)*norm
    
    num=0
    denom=0
    #inv_sigma_squared = 0

    for i, f in enumerate(freq): 

      if i == 0: continue # Skipping DC offset

      #inv_sigma_squared += (2*2*T*np.abs(s_fft[i])**2/J[i]) 
      
      num += np.exp(2j*np.pi*f*t0)*np.conjugate(s_fft[i])*v_fft[i]/J[i]
      denom += np.abs(s_fft[i])**2/J[i]
        
    A = num/denom
    #sigma = np.sqrt(1/inv_sigma_squared)
    
    return A #, sigma


def get_noise(noise_count):
  
  '''This function opens up the correct noise library and pulls out noise_count many traces from it.'''

  file = up.open("../nexo_offline_waveform_study/expected_1_2_us_noise_lib_100e.root")
  noise = np.asarray(file['noiselib']['noise_int'])[0:noise_count]
  del file # Deleting the file after we save our traces to keep memory down
  return noise  

def get_noise_new(noise_count):
  file = up.open("/Users/yumiao/Documents/Works/0nbb/nEXO/offline-samples/noise_lib_1_2_us_100e.root")
  tree = file['noiselib']
  noise_vec_load = tree.arrays(entry_start=0, entry_stop=noise_count, library='np')
  noise = noise_vec_load['noise_int']
  noise = np.vstack(noise)
  return noise
  


def add_noise(truth, scale=1):
  length = truth.size
  full_noise = get_noise_new(10000) # Choosing high noise count to ensure randomness
  #full_noise = get_noise(10000) # Choosing high noise count to ensure randomness
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

  Z_anode = 403 #mm
  lifetime = 1e4 #us
  velocity = 1.71 #mm/us
  correction  = np.exp((np.abs(Z)-Z_anode)/(lifetime*velocity)) 

  collection = electrons*(1-noise_tag)

  channel_count = noise_tag.size # get the number of channels in the event and their length

  event_reconstruction = np.zeros(channel_count) # Intialzing array to hold channel reconstruction values before summing for event reocnstruciton 
  event_error = np.zeros(channel_count)

  for c, unscaled_truth in enumerate(std_truth):
 
    print('---'*10)
    print('Channel number: ', c)
    
    # Cut channels that are noise
    if noise_tag[c] == 1: 
      print('Noise channel')
      continue 

    if collection[c] == 0:
      print('Ion channel - no true signal observed.')
      continue

    if collection[c] <= cut:
      print('Waveform below threshold- true colleciton is {0} e^-: Adding this to reconstruction for channel.'.format(collection[c]))
      event_reconstruction[c] = -1*collection[c] # Adding true collection to reconstruction error for channels below threshold - truth info here to avoid ion effect 
      continue

    # Putting the factor of 9 back into the data since uproot won't allow it until this point
    scale = 9#/3.03
    truth = scale*np.asarray(unscaled_truth)
    channel = add_noise(truth, scale = 1.04)
    print('This channel collected {0} electrons'.format(collection[c]))
    print('But the integral of the true WVFM is {0}'.format(np.sum(truth)))

    try: unit = truth/np.max(truth)
    except: 
      print('Stencil could not be created - most likely incomplete file.')
      print('Removing this event from analysis and continuing.')
      event_reconstruction = np.full(len(event_reconstruction), np.nan)
      return event_reconstruction
    
    A= optimal_filter(channel, unit, J, 0)
    recon_amp = np.real(A)
    print('Reconstructed amplitude is {0}'.format(recon_amp))
    print('True amplitude of the signal is {0}'.format(np.max(truth)))
    print('Area of the unit pulse stencil is {0}'.format(np.sum(unit)))
    event_reconstruction[c] = recon_amp*np.sum(unit) - np.sum(truth) # Subtracting integral of true waveform here to counteract ion effect in reconstruction
    #event_error[c] = sigma**2
    print('The correction factor for this event is {0}'.format(correction))
    #print('Error on channel reconstruction is ', event_reconstruction[c]*correction)
    #print('Predicted error for this channel is ', np.sqrt(np.sum(event_error)))

  return event_reconstruction*correction, 


def save_data(reconstruction, job_num):
  current_run_file  = len(os.popen('ls -d ./jobs/*/').read().split('\n'))-1
  file_name = './data/Run_'+str(current_run_file)+'/nEXO_Charge_analysis_file_' +str(job_num) + '.npy'
  np.save(file_name, reconstruction, allow_pickle=True)


def start_job(job_num, cut):

  file_list = glob.glob('../pad_lib_redownload/*.root')
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
  error_limit = np.zeros(event_count)

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

    data.get_event_electrons()
    data.get_E()
    data.get_lineage()

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

    #if np.abs(Z) > Z_anode + 1*(Z_cathode-Z_anode) or np.abs(Z) < Z_anode + 0.9*(Z_cathode-Z_anode):
      #print('Event outside of region of interest')
      #reconstruction[e] = np.nan
      #continue

    #Brem cut:
    #if np.any(data.lineage==7):
    #  print('Event {0} is a Brem event - cutting it from analysis')
    #  reconstruction[e] = np.nan
    #  continue
    
    # Pull noise from library to generate PSD
    N = len(truth[0])
    noise_traces = get_noise(10000)[0:N,:]
    noise_FFT = np.abs(np.fft.rfft(noise_traces, axis = 1))**2
    Fs = 2e6
    norm = 2/(N*Fs)
    J = np.mean(noise_FFT, axis = 0)*norm
    
    found_electrons = event_analysis(truth, J, Z, noise_tag, electrons, cut)

    reconstruction[e] = np.sum(found_electrons)

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
