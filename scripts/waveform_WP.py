import numpy as np
import ROOT
from array import array

from scripts.nEXOFieldWP import nEXOFieldWP
from scripts.toy_digitization import digitization

class waveform_WP:
    
    def __init__(self):
        self.digi = digitization(SamplingFrequency=2.0)
        
        self.wp = nEXOFieldWP()
        self.fSamplingSeqZ = []
        self.fZSeqTemplate = None
        
        self.fPCDZ = None
        
        self.v_drift = 1.74987 # mm / us
        self.sampling_interval = 0.5 # us
        self.fAnodeZ = -402.97 # unit: mm
        
        self.onechannel_time = None
        self.onechannel_wf = None
        self.onechannel_time_pointcharge = None
        self.onechannel_wf_pointcharge = None
        self.onechannel_curtime_pointcharge = None
        self.onechannel_curwf_pointcharge = None
        self.strip_NTE = 0.
        self.strip_Qion = 0.

        self.set_SamplingZSeq = False

    def initialize(self):
        self.wp._load_axes()
        self.initialize_ZSeq()
        # Will not do this until we have a z-distribution of the PCDs.
        #self.initialize_PCDZ()

    def initialize_ZSeq(self):
        '''
        # This is mannually defined at long times, would be nice to replace these with a formula,
        # ... but seem to be magic for not getting wiggles in ASIC response!
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
        '''
        tmax, tmin = 1500., 20.
        n = 50
        ZBinsVec = [0 for i in range(n)]
        for i in range(n):
            l = i/(n+1.)
            ZBinsVec[i] = np.exp((1-l)*np.log(tmax) + l*np.log(tmin) ) 
        # linear spaced fine sampling at short distances
        dZ = ZBinsVec[-2] - ZBinsVec[-1]
        while ZBinsVec[-1] > 1.3*dZ:
            ZBinsVec.append(ZBinsVec[-1] - dZ)
        ZBinsVec.append(0.)
        
        # reverse to ascending order
        ZBinsVec = ZBinsVec[::-1]
        ZBinsVec_array = array('d', ZBinsVec)
        self.fZSeqTemplate = ROOT.TAxis(len(ZBinsVec)-1, ZBinsVec_array)


    
    def initialize_PCDZ(self, Zseq):
        if not self.fPCDZ:
            dZ = self.v_drift * self.sampling_interval
            self.fPCDZ = ROOT.TH1F('pcdZ', '', 10000, self.fAnodeZ - 1500.*dZ, self.fAnodeZ)
        else:
            self.fPCDZ.Reset()
        
        for zelem in Zseq:
            self.fPCDZ.Fill(zelem)

        
    
    def DetermineSamplingSequence(self, maxZ, minZ):
        # minZ < maxZ > fAnodeZ
        # sampling sequence is for fAnodeZ -> maxZ -> minZ (reverse sign order), relative to fAnodeZ
        # producing (0, .   .  .  . ... fAnodeZ -maxZ . ..  . ..fAnodeZ - minZ) 
        # points starting from anode = 0, approahcing maxZ following template sequence spacing
        self.fSamplingSeqZ = []
        self.fSamplingSeqZ.append(0.)
        anode_bin = self.fZSeqTemplate.FindBin(self.fAnodeZ - maxZ) 
        while anode_bin >= 1:
            self.fSamplingSeqZ.append(self.fAnodeZ - maxZ - self.fZSeqTemplate.GetBinLowEdge(anode_bin))
            anode_bin = anode_bin - 1
        currentBin  = self.fPCDZ.FindBin(maxZ)
        minBin      = self.fPCDZ.FindBin(minZ)

        while currentBin > minBin:
            b1 = currentBin - 1
            while b1 > 0:
                if self.fPCDZ.GetBinContent(b1) :
                    break
                b1 = b1 - 1
        
            deltaZ = self.fPCDZ.GetBinCenter(currentBin) - self.fPCDZ.GetBinCenter(b1)
            Zfinal = self.fSamplingSeqZ[-1] + deltaZ
            bin = self.fZSeqTemplate.FindBin(deltaZ)
            while bin >= 1:
                self.fSamplingSeqZ.append(Zfinal - self.fZSeqTemplate.GetBinLowEdge(bin))
                bin = bin - 1
            currentBin = b1

        print(f'Diffused charges distribute between {minZ} to {maxZ} with {len(self.fSamplingSeqZ)} sampling points.')


    def initialize_samplingZSeq(self, samplingSeqZ):
        self.fSamplingSeqZ = samplingSeqZ
        self.set_SamplingZSeq = True

    
        
    def IsPointChargeOnStrip(self, dX, dY, IsXStrip=True):
        PadSize = 6.0
        HalfPadSize = PadSize / 2.0
        dX, dY = np.abs(dX), np.abs(dY)
        if not IsXStrip:
           dX0, dY0 = dX, dY
           dX, dY = dY0, dX0
        dx_a, dy_a = dX % PadSize, dY % PadSize
        #if dY < 48. and dX < HalfPadSize and ( (dy_a < (HalfPadSize - dx_a)) or (dy_a > (HalfPadSize + dx_a)) ):
        if dY < 48. and dX < HalfPadSize and ( (dy_a < (HalfPadSize - dX)) or (dy_a > (HalfPadSize + dX)) ):
            return True
        else:   
            return False 
        

    def InterpolateWaveform(self, dX, dY, z):
        amplitude = 0.
        if z >= 0.05:
            amplitude = self.wp.interpolate(z)
        elif 0.0 < z < 0.05:
            amplitude = self.wp.interpolate(0.05)
        else:
            if self.IsPointChargeOnStrip(dX, dY):
                amplitude = 1e5
            else:
                amplitude = 0.
        return amplitude
        
        
            
    def CalcPointChargeWaveformOnChannel(self, dX, dY, iniZ, Q):
        ## The input dX and dY are the distance from the charge to the channel center in mm.
        ## The axis rotation should be done before if the channel is a y-strip.
        ## The iniZ is the absolute distance from the charge to the anode in mm.
        PadSize = 6.0
        HalfPadSize = PadSize / 2.0
        wf = np.zeros(500)
        xId, yId = self.wp.GetXBin(dX), self.wp.GetYBin(dY)
        if xId > 300 or yId > 300:
            print(f'The charge is too far from the strip ({dX:.2f} mm, {dY:.2f} mm).')
            return
        #dx_a, dy_a = dX % PadSize, dY % PadSize
        self.wp.GetHist(xId, yId)
        qIon = self.wp.interpolate(iniZ) # Ion is assumed to be not moving.
        totalIonQ = qIon / 1e5 * Q
        NTE = 0.
        #if (dY < 48. and dX < HalfPadSize and (dy_a < (HalfPadSize - dx_a) or (dy_a > (HalfPadSize + dx_a)))):
        if self.IsPointChargeOnStrip(dX, dY):
            NTE = Q
        for j in range(len(self.fSamplingSeqZ)):
            deltaZ = self.fSamplingSeqZ[j]
            zj = iniZ - deltaZ
            amplitude = self.InterpolateWaveform(dX, dY, zj)
            wf[j] = (amplitude - qIon)/1e5 * Q

        self.onechannel_time_pointcharge, self.onechannel_wf_pointcharge = [], []
        samplingTime = 0.
        for j in range(len(self.fSamplingSeqZ)):
            samplingTime = self.fSamplingSeqZ[j] / (self.v_drift)
            self.onechannel_time_pointcharge.append(samplingTime)
            self.onechannel_wf_pointcharge.append(wf[j])    
        # Fill the last two points:
        self.onechannel_time_pointcharge.append(samplingTime + 0.5)
        lasttime = max(samplingTime+20, 1500.*0.5)
        self.onechannel_time_pointcharge.append(lasttime)
        self.onechannel_wf_pointcharge.append(NTE - totalIonQ)    
        self.onechannel_wf_pointcharge.append(NTE - totalIonQ)    
        self.strip_NTE = NTE
        self.strip_Qion = totalIonQ

        self.onechannel_time_pointcharge = np.array(self.onechannel_time_pointcharge)
        self.onechannel_wf_pointcharge = np.array(self.onechannel_wf_pointcharge)
                


    def CalcWaveformOnChannel(self, strip_x, strip_y, strip_type, deposits):
        PadSize = 6.0
        HalfPadSize = PadSize / 2.0
        wf = np.zeros(500)
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
        
    def quantize_pointCharge_waveform(self):
        self.digi.convolve_asic_response(self.onechannel_time_pointcharge, self.onechannel_wf_pointcharge)
        self.digi.quantization_trueWF(self.digi.cryoAmp, 40000.)
        self.onechannel_curwf_pointcharge = self.digi.fTruth
        self.onechannel_curtime_pointcharge = np.arange(0, len(self.onechannel_curwf_pointcharge), 1) * 0.5
        