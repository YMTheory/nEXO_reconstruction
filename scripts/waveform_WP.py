import numpy as np
import math

from scripts.nEXOFieldWP import nEXOFieldWP
import ROOT

class waveform_WP:
    
    def __init__(self):
        self.wp = nEXOFieldWP()
        self.fSamplingSeqZ = []
        self.fZSeqTemplate = None
        
        self.fPCDZ = None
        
        self.v_drift = 1.7 # mm / us
        self.sampling_interval = 0.5 # us
        
        self.onechannel_time = None
        self.onechannel_wf = None

    def initialize(self):
        self.wp._load_axes()
        self.initialize_PCDZ()
        self.initialize_ZSeq()

    def initialize_ZSeq(self):
        ZbinsVec = [1350., 1300., 1100., 900., 700., 500., 300., 200., 170., 150., 130., 120., 110., 100., 90., 80., 75., 70., 65., 60., 55., 50., 45., 40., 35., 30., 25., 24., 23., 22., 21., 20. ]
        dZ = self.v_drift * self.sampling_interval # mm
        tmpZ = 20.
        while tmpZ-dZ > 0:
            ZbinsVec.append(tmpZ-dZ)
            tmpZ = tmpZ - dZ
        ZbinsVec.append(0.)
        Nbin = len(ZbinsVec)
        Zbins = np.zeros(200)
        for i in range(Nbin):
            Zbins[i] = ZbinsVec[Nbin-1-i]
        self.fZSeqTemplate = ROOT.TH1F('ZSeqTemplate', '', Nbin-1, Zbins) 
        # fZSeqTemplate saves the delta_Z values.

    
    def initialize_PCDZ(self):
        anodeZ = -402.97 #mm
        dZ = self.v_drift * self.sampling_interval # mm
        self.fPCDZ = ROOT.TH1F('PCDZ', '', 10000, anodeZ-dZ*1500, anodeZ)
    
    
    def DetermineSamplingSequence(self, maxZ, minZ):
        anodeZ = -402.97 #mm
        self.fSamplingSeqZ = []
        self.fSamplingSeqZ.append(maxZ)
        bin = self.fZSeqTemplate.FindBin(anodeZ - maxZ)
        
        i = bin
        while i >= 1:
            self.fSamplingSeqZ.append(anodeZ - self.fZSeqTemplate.GetBinLowEdge(i) )
            i = i-1
        # self.fSamplingSeqZ, from maxZ coordinate to anodeZ coordinate, in mm
        
        # determine sampling points between minZ and maxZ
        currentBin = self.fPCDZ.FindBin(maxZ)
        minBin = self.fPCDZ.FindBin(minZ)
        deltaZ = 0.
        while currentBin > minBin:
            idx = 0
            j = currentBin - 1
            while idx == 0:
                idx = j
                deltaZ = self.fPCDZ.GetBinCenter(currentBin) - self.fPCDZ.GetBinCenter(j)
                j = j-1
            currentZ = self.fSamplingSeqZ[len(self.fSamplingSeqZ)-1]
            bin = self.fZSeqTemplate.FindBin(deltaZ)
            for i in range(bin):
                j = bin - i
                self.fSamplingSeqZ.append( currentZ + deltaZ - self.fZSeqTemplate.GetBinLowEdge(j) )
            currentBin = idx
        
        for i in range(len(self.fSamplingSeqZ)):
            deltaZ = self.fSamplingSeqZ[i] - maxZ
            self.fSamplingSeqZ[i] = deltaZ
            # convert to distance from maxZ
            

    def CalcWaveformOnChannel(self, strip_x, strip_y, strip_type, deposits):
        PadSize = 6.0
        wf = np.zeros(len(self.fSamplingSeqZ))
        totalIonQ = 0.0
        for depo in deposits:
            x, y, zinit, q = depo['x'], depo['y'], depo['z'], depo['q']
            if strip_type == 'xstrip':
                dx, dy = np.abs(strip_x - x), np.abs(strip_y - y)

            elif strip_type == 'ystrip':
                dy, dx = np.abs(strip_x - x), np.abs(strip_y - y)

            else:
                print('Strip type not recognized.')
                return

            xId, yId = self.wp.GetXBin(dx), self.wp.GetYBin(dy)
            
            dx_a = dx % PadSize
            dy_a = dy % PadSize

            print(f'(dx, dy) = ({dx}, {dy}), (dx_a, dy_a) = ({dx_a}, {dy_a}).')

            if xId > 300 or yId > 300:
                continue # Too far from the strip
            
            self.wp.GetHist(xId, yId)
            qIon = self.wp.interpolate(zinit)
            totalIonQ += qIon / 1e5 * q

            NTE = 0
            for j in range(len(self.fSamplingSeqZ)):
                deltaZ = self.fSamplingSeqZ[j]
                amplitude = 0.
                if zinit-deltaZ > 0.0:
                    amplitude = self.wp.interpolate( zinit - deltaZ)
                else:
                    if (dy<48 and dx<PadSize and dy_a>dx_a and dy_a<(PadSize-dx_a)):
                        amplitude = 1e5
                        NTE = q
                    else:
                        amplitude = 0.0

                wf[j] = (amplitude - qIon) / 1e5 * q
                
        samplingTime = 0.
        self.onechannel_time = []
        self.onechannel_wf = []
        for j in range(len(self.fSamplingSeqZ)):
            samplingTime = self.fSamplingSeqZ[j] / (self.v_drift)    
            self.onechannel_time.append(samplingTime)
            self.onechannel_wf.append(wf[j])
        # Fill the last two points:
        self.onechannel_time.append( samplingTime + 0.5)          
        fSimulationTime = 1500. * 0.5
        lasttime = max(samplingTime+20, fSimulationTime)
        self.onechannel_time.append(lasttime)
        self.onechannel_wf.append(NTE-totalIonQ)
        self.onechannel_wf.append(NTE-totalIonQ)
        