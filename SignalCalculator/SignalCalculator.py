import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = (10,8)


# This is the function called "f" in Mike's paper
def ImageChargeDefiniteIntegral(x,y,z):
    return np.arctan(x*y / (z * np.sqrt(x**2 + y**2 + z**2)))


######################################################################################\
# Define a function to compute the charge observed by a square pad, centered at (0,0,0) 
def InducedChargeSquarePadNumerical(Q,x,y,z,padSize=3.,numPoints=300):
    
    padDiagonalSize = padSize # mm
    padEdgeSize = padDiagonalSize/np.sqrt(2.)
    
    
    xarray = np.linspace(-padEdgeSize/2.,padEdgeSize/2.,numPoints)

    pointSpacing = xarray[2]-xarray[1]

    xv,yv = np.meshgrid(xarray,xarray)
    
    chargeDensity =  Q * z / (2 * np.pi * ((x-xv)**2 + (y-yv)**2 + z**2)**(3/2))
    
    induced_charge = np.sum(chargeDensity * pointSpacing**2 )
    
    if z > 0.:
        return induced_charge
    if z == 0.:
        if (x>xarray[0])&(x<xarray[-1])&(y>xarray[0])&(y<xarray[-1]):
            return Q
        else:
            return 0.
        
        
######################################################################################\
# Define a function to compute the charge observed by a square pad, centered at (0,0,0) 
def InducedChargeSquarePadExact(Q,x,y,z,padSize=3.):
    
    padDiagonalSize = padSize # mm
    padEdgeSize = padDiagonalSize/np.sqrt(2.)
    
    induced_charge = Q /(2.*np.pi) * ( \
                        ImageChargeDefiniteIntegral(-padEdgeSize/2.-x,-padEdgeSize/2.-y,z)\
                        - ImageChargeDefiniteIntegral(padEdgeSize/2.-x,-padEdgeSize/2.-y,z)\
                        - ImageChargeDefiniteIntegral(-padEdgeSize/2.-x,padEdgeSize/2.-y,z)\
                        + ImageChargeDefiniteIntegral(padEdgeSize/2.-x,padEdgeSize/2.-y,z))
    
    
    if z > 0.:
        return induced_charge
    if z == 0.:
        if (x>-padEdgeSize/2.)&(x<padEdgeSize/2.)&(y>-padEdgeSize/2.)&(y<padEdgeSize/2.):
            return Q
        else:
            return 0.
     
######################################################################################\
# Define a function to compute the charge observed by a square pad, centered at (0,0,0) 
def InducedChargeSquarePadExactVectorized(Q,x,y,z,padSize=3.):
    
    padDiagonalSize = padSize # mm
    padEdgeSize = padDiagonalSize/np.sqrt(2.)
    
    induced_charge = Q /(2.*np.pi) * ( \
                        ImageChargeDefiniteIntegral(-padEdgeSize/2.-x,-padEdgeSize/2.-y,z)\
                        - ImageChargeDefiniteIntegral(padEdgeSize/2.-x,-padEdgeSize/2.-y,z)\
                        - ImageChargeDefiniteIntegral(-padEdgeSize/2.-x,padEdgeSize/2.-y,z)\
                        + ImageChargeDefiniteIntegral(padEdgeSize/2.-x,padEdgeSize/2.-y,z))
    
    zmask = z > 0.
    return_array = np.zeros(len(induced_charge))
    return_array[zmask] = induced_charge[zmask]

    on_strip_mask = (x>-padEdgeSize/2.)&(x<padEdgeSize/2.)&(y>-padEdgeSize/2.)&(y<padEdgeSize/2.)

    return_array[np.invert(zmask)&on_strip_mask] = np.ones( len(return_array[np.invert(zmask)&on_strip_mask]) ) * \
                                                           Q[np.invert(zmask)&on_strip_mask]
    return_array[np.invert(zmask)&np.invert(on_strip_mask)] = np.zeros (len(return_array[np.invert(zmask)&np.invert(on_strip_mask)]) ) 
    
    return return_array

        
######################################################################################\
def InducedChargeNEXOStrip(Q,x,y,z,padSize=6.,numPads=16):
    # Strip is modeled as a strip along the X axis, with half the strip above 0 and 
    # half below
    totalInducedCharge = 0.
    stripLength = padSize * numPads
    
    for i in range(numPads):
        xpad = -stripLength/2. + padSize/2. + padSize * i
        
        #print(xpad)
        
        relativeX = (x-xpad)/np.sqrt(2.) + y/np.sqrt(2.)
        relativeY = -(x-xpad)/np.sqrt(2.) + y/np.sqrt(2.)
        
        #print('{}\t{:4.4},{:4.4}'.format(xpad,relativeX,relativeY))
        
        
        totalInducedCharge += InducedChargeSquarePadExact(Q,relativeX,relativeY,z,\
                                                     padSize)
        
    return totalInducedCharge


        
######################################################################################\
def InducedChargeNEXOStripVectorized(Q,x,y,z,padSize=6.,numPads=16):
    # Strip is modeled as a strip along the X axis, with half the strip above 0 and 
    # half below
    totalInducedCharge = np.zeros(len(x))
    stripLength = padSize * numPads
    
    for i in range(numPads):
        xpad = -stripLength/2. + padSize/2. + padSize * i
        
        #print(xpad)
        
        relativeX = (x-xpad)/np.sqrt(2.) + y/np.sqrt(2.)
        relativeY = -(x-xpad)/np.sqrt(2.) + y/np.sqrt(2.)
        
        #print('{}\t{:4.4},{:4.4}'.format(xpad,relativeX,relativeY))
        
        
        totalInducedCharge += InducedChargeSquarePadExactVectorized(Q,relativeX,relativeY,z,\
                                                     padSize)
        
    return totalInducedCharge
        


######################################################################################\    
def ComputeChargeWaveformOnStrip(Q,x,y,z,padSize=6.,numPads=16,numWfmPoints=300):
    
    driftVelocity = 1.7 # mm/us
    
    zpoints = np.linspace(0.,z,numWfmPoints)
    qpoints = np.ones(numWfmPoints)
    
    for i in range(numWfmPoints):
        qpoints[i] = InducedChargeNEXOStrip(Q,x,y,zpoints[i],\
                                            padSize,numPads)
        
        
    driftpoints = zpoints/driftVelocity
    
    return driftpoints,np.flip(qpoints)

######################################################################################\    
def ComputeChargeWaveformOnStripWithIons(Q,x,y,z,padSize=6.,numPads=16,numWfmPoints=300):
    
    driftVelocity = 1.7 # mm/us
    
    zpoints = np.linspace(0.,z,numWfmPoints)
    qpoints = np.ones(numWfmPoints)
    
    for i in range(numWfmPoints):
        qpoints[i] = InducedChargeNEXOStrip(Q,x,y,zpoints[i],\
                                            padSize,numPads) + \
                     InducedChargeNEXOStrip(-Q,x,y,zpoints[-1],\
                                            padSize,numPads)  
        
        
    driftpoints = zpoints/driftVelocity

    #return driftpoints,np.flip(qpoints)

    #### Adding by Miao currently...
    qpoints = np.flip(qpoints)

    driftpoints = np.append(driftpoints, driftpoints[-1]+20)
    qpoints     = np.append(qpoints, qpoints[-1])

    return driftpoints, qpoints
    

######################################################################################\    
def ComputeCurrentWaveformOnStrip(Q,x,y,z,padSize=6.,numPads=16,numWfmPoints=300):
    
    driftVelocity = 1.7 # mm/us
    
    zpoints = np.linspace(0.,z,numWfmPoints)
    ipoints = np.ones(numWfmPoints)
    
    zstep = zpoints[2]-zpoints[1]
    
    for i in range(numWfmPoints):
        qpointlo = InducedChargeNEXOStrip(Q,x,y,zpoints[i],\
                                            padSize,numPads)
        qpointhi = InducedChargeNEXOStrip(Q,x,y,zpoints[i]+zstep/10.,\
                                            padSize,numPads)
        ipoints[i] = (qpointlo-qpointhi)/(zstep/10./driftVelocity)
        
    driftpoints = zpoints/driftVelocity
    
    return driftpoints,np.flip(ipoints)
   
######################################################################################
def DrawStrip( padSize=6., numPads=16 ):
    xstart = -(numPads*padSize)/2.
    
    rectangles = []
    
    for i in range(numPads):
        rectangles.append( plt.Rectangle( (xstart+padSize*i + padSize/2., -padSize/2.), \
                                         padSize/np.sqrt(2), padSize/np.sqrt(2), \
                                  fc=(0.,0.,0.,0.2),ec=(0.,0.,0.,1.),angle=45. ) )
        plt.gca().add_patch(rectangles[i])
    

    plt.xlim(xstart-padSize,-1*(xstart-padSize))
    plt.ylim(xstart-padSize,-1*(xstart-padSize))


