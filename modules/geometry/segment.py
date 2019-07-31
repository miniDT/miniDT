"""Module with classes to represent segments"""

import numpy as np
from math import asin
from pdb import set_trace as br
from modules.geometry import Geometry, DTYPE_COOR

class Segment:
    """Represents a segment in 3D space"""

    def __init__(self, start, end):
        self.start = np.array(start, dtype=DTYPE_COOR)
        self.end = np.array(end, dtype=DTYPE_COOR)


    def calc_vector(self):
        """Calculates the vector representation of the segment"""
        self.origin = np.array(self.start)
        self.vector = self.end - self.start
        self.vector /= np.linalg.norm(self.vector)


    def toSL(self, SL):
        """Creates a new segment with coordinates converted to the local reference frame of SL"""
        start = SL.coor_to_local(self.start.reshape((1, 3)))[0]
        end = SL.coor_to_local(self.end.reshape((1, 3)))[0]
        return Segment(start, end)

    
    def fromSL(self, SL):
        """Creates a new segment with coordinates converted from the local reference frame of SL"""
        start = SL.coor_to_global(self.start.reshape((1, 3)))[0]
        end = SL.coor_to_global(self.end.reshape((1, 3)))[0]
        return Segment(start, end)

    def pointAtZ(self, z):
        """Returns coordinates of intersection with XY plane at specified Z coordinate"""
        t = (z - self.origin[2]) / self.vector[2]
        return self.origin + self.vector * t
