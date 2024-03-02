import numpy as np

from scripts.nEXOCoordinateKey import nEXOCoordinateKey


class grouper():
    def __init__(self) -> None:
        self.fPCDMaps           = {}
        self.fInductionPCDMaps  = {}
        
    
        
    def AddTEPoint(self, x, y, z, q, keyType):
        k = nEXOCoordinateKey((x, y, z), 0., keyType)
        if k not in self.fPCDMaps:
            self.fPCDMaps[k] = q
        else:
            self.fPCDMaps[k] += q


    def generatePCDs(self, grid_x, grid_y, grid_z, grid_q):
        self.fPCDMaps = {}

        for i in range(len(grid_x)):
            xi, yi, zi, qi = grid_x[i], grid_y[i], grid_z[i], grid_q[i]
            self.AddTEPoint(xi, yi, zi, qi, 0)
            
            

    def GetPCDInfo(self):
        pcd_x, pcd_y, pcd_z, pcd_q = [], [], [], []
        for k, v in self.fPCDMaps.items():
            x, y, z, t = k.GetCenter()
            pcd_x.append(x)
            pcd_y.append(y)
            pcd_z.append(z)
            pcd_q.append(v)
            
        pcd_x = np.array(pcd_x)
        pcd_y = np.array(pcd_y)
        pcd_z = np.array(pcd_z) 
        pcd_q = np.array(pcd_q)
        
        return pcd_x, pcd_y, pcd_z, pcd_q