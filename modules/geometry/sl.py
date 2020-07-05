"""Module handling a single superlayer (chamber)"""

import numpy as np
import math
from math import cos, sin
from pdb import set_trace as br
from modules.geometry import DTYPE_COOR

class SL:
    """Superlayer with all the properties for coordinate transformations"""

    id = -1
    trans_matrix = np.array([       # affine transformation matrix wrt to the global reference frame
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=DTYPE_COOR)

    def __init__(self, index, shift=(0,0,0), rotation=(0,0,0)):
        self.id = index
        # self.rf = CoordSys3D('SL_{0:d}'.format(self.id))
        # self.plane = Plane(p1=Point3D(0,0,0), normal_vector=(0,0,1))
        self.set_shift(shift)
        self.set_rotation(rotation)
        # self.update_rf()
        # self.point = self.rf.origin.locate_new('point', self.S['p'][0]*self.rf.i + self.S['p'][1]*self.rf.j + self.S['p'][2]*self.rf.k)

    def set_shift(self, shift=(0,0,0)):
        """Update the shift of the reference frame in the transformation matrix"""
        for i in range(3):
            self.trans_matrix[i][3] = shift[i]

    def set_rotation(self, rot=(0,0,0)):
        """Update the rotation of the reference frame in the transformation matrix"""
        # Rotation matrix around the Z axis
        rZ = np.array([
            [cos(rot[2]), -sin(rot[2]), 0],
            [sin(rot[2]),  cos(rot[2]), 0],
            [0,            0,           1]
        ], dtype=DTYPE_COOR)
        # Rotation matrix around the Y axis
        rY = np.array([
            [ cos(rot[1]), 0, sin(rot[1])],
            [ 0,           1,           0],
            [-sin(rot[1]), 0, cos(rot[1])]
        ], dtype=DTYPE_COOR)
        # Rotation matrix around the X axis
        rX = np.array([
            [1,           0,            0],
            [0, cos(rot[0]), -sin(rot[0])],
            [0, sin(rot[0]),  cos(rot[0])]
        ], dtype=DTYPE_COOR)
        rM = rZ.dot(rY).dot(rX)
        # Adding missing rows and columns
        rM = np.concatenate((rM, np.zeros((1,3), dtype=DTYPE_COOR)))
        rM = np.hstack((rM, self.trans_matrix[:, 3].reshape((4,1))))
        self.trans_matrix = rM

    def coor_to_global(self, pos_xyz_local):
        """Converts 2D array of local coordinates to coordinates in the global reference frame"""
        n_hits = len(pos_xyz_local)
        # Adding Y coordinates set to 0.0 and 4-th component set to 1 for point representation
        pos_local = np.hstack((pos_xyz_local, np.ones((n_hits, 1), dtype=DTYPE_COOR)))
        pos_global = np.transpose(self.trans_matrix.dot(np.transpose(pos_local)))
        return pos_global[:, :3]

    def coor_to_local(self, pos_xyz_global):
        """Converts 2D array of global coordinates to coordinates in the local reference frame"""
        n_hits = len(pos_xyz_global)
        pos_global = np.hstack((pos_xyz_global, np.ones((n_hits, 1), dtype=DTYPE_COOR)))
        pos_local = np.transpose(np.linalg.inv(self.trans_matrix).dot(np.transpose(pos_global)))
        return pos_local[:, :3]





if __name__ == '__main__':
    sls = {
        0: SL(0, (500, 0, 0)),
        1: SL(1, (0, 100, 50), (0, 0, 0.5*math.pi)),
    }

    s = sls[0]

