import numpy as np

class nEXOCoordinateKey():
    def __init__(self, pos, t, keyType) -> None:
        self.fMCChargePixelSizeXY   = 0.
        self.fMCChargePixelSizeZ    = 0.
        self.fMCChargePixelTime     = 0.
        self.fUorX                  = 0.
        self.fVorY                  = 0.
        self.fZ                     = 0.
        self.fT                     = 0.

        self.SetKeyType(keyType)
        self.SetCoordinates(pos, t)

    def __eq__(self, other) -> bool:
        if ( other.fUorX == self.fUorX and other.fVorY == self.fVorY and other.fZ == self.fZ and other.fT == self.fT):
            return True
        return False

    def __hash__(self) -> int:
        return hash((self.fUorX, self.fVorY, self.fZ, self.fT))
        
    def SetKeyType(self, keyType):
        fKeyType = keyType
        
        if fKeyType == 0:
            self.fMCChargePixelSizeXY = 1.5*np.sqrt(2)
            self.fMCChargePixelSizeZ = 0.75
            
        elif fKeyType == 1:
            self.fMCChargePixelSizeXY = 1.5*np.sqrt(2)
            self.fMCChargePixelSizeZ = 0.75

        self.fMCChargePixelTime = 0.5
        
        
    def SetCoordinates(self, pos, t):
        x, y, z = pos
        self.fUorX = int ((x + y)/2.*np.sqrt(2) / self.fMCChargePixelSizeXY)
        if (x + y ) < 0:
            self.fUorX = self.fUorX - 1
        self.fVorY = int((y-x) /2.*np.sqrt(2)/self.fMCChargePixelSizeXY)
        if (y - x) < 0:
            self.fVorY = self.fVorY - 1
        
        self.fZ = int(z / self.fMCChargePixelSizeZ)
        if z < 0:
            self.fZ = self.fZ - 1
        self.fT = int(t / self.fMCChargePixelTime)
        if t < 0:
            self.fT = self.fT - 1
            
            
     
    def GetCenter(self):
        UorX = self.fMCChargePixelSizeXY * (0.5 + self.fUorX)
        VorY = self.fMCChargePixelSizeXY * (0.5 + self.fVorY)
        Z    = self.fMCChargePixelSizeZ * (0.5 + self.fZ)
        T    = self.fMCChargePixelTime * (0.5 + self.fT)
        
        cos = np.cos(np.pi/4.)
        sin = np.sin(np.pi/4.)
        x_rot = UorX * cos - VorY * sin
        y_rot = UorX * sin + VorY * cos
        
        return (x_rot, y_rot, Z, T)