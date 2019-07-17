"""Module with classes for hits"""

import pandas as pd
import numpy as np
from pdb import set_trace as br
from modules.geometry import Geometry, DTYPE_COOR
from modules.analysis import config

class ProtoHit:
    """Hit as seen by the DAQ"""

    def __init__(self, sl, layer, wire, time):
        self.sl = sl
        self.layer = layer
        self.wire = wire
        self.time = time


class Hit:
    """Spatial representation of a hit as used in the reconstruction"""

    x = 0.0

    def __init__(self, protoHit, side):
        self.protoHit = protoHit
        self.side = side

    @classmethod
    def fromProtoHit(cls, protoHit):
        """Returns a list of two hits (left + right) from a single ProtoHit"""
        return [cls(protoHit, -1), cls(protoHit, 1)]


class HitManager:
    """Manages hits stored inside a Pandas dataframe"""
    hits = None
    HIT_DTYPES = {'sl': np.uint8, 'layer': np.uint8, 'wire': np.uint8, 'time': np.float32}
    POS_RES = (0.3, 0.0, 0.0)

    def __init__(self):
        self.G = Geometry(config)
        self.reset()

    def add_hits(self, hits_list):
        """Adds a hit with the given parameters"""
        hits = pd.DataFrame(hits_list, columns=['sl', 'layer', 'wire', 'time']).astype(self.HIT_DTYPES)
        self.hits = self.hits.append(hits, ignore_index=True)

    def reset(self):
        """Removes all hits from the dataframe"""
        self.hits = pd.DataFrame(columns=['sl', 'layer', 'wire', 'time']).astype(self.HIT_DTYPES)

    def calc_pos(self, sls=None):
        """Calculates the left/right hit positions for all the hits in the dataframe"""
        # Filling the hit positions in the local reference frame of the superlayer
        self.G.fill_hits_position(self.hits, reco=True)
        if sls:
            nhits = len(self.hits)
            zeros = np.zeros(nhits, dtype=DTYPE_COOR)
            for col in ['glposx', 'glposy', 'glposz', 'grposx', 'grposy', 'grposz']:
                self.hits[col] = zeros
            # Calculating the hit positions for each SL in the global reference frame
            for iSL, sl in sls.items():
                sel = self.hits['sl'] == iSL
                # Updating global positions for left hits
                pos_global = sl.coor_to_global(self.hits.loc[sel, ['lposx', 'posy', 'posz']].values)
                self.hits.loc[sel, ['glposx', 'glposy', 'glposz']] = pos_global
                # Updating global positions for right hits
                pos_global = sl.coor_to_global(self.hits.loc[sel, ['rposx', 'posy', 'posz']].values)
                self.hits.loc[sel, ['grposx', 'grposy', 'grposz']] = pos_global


if __name__ == '__main__':
    H = HitManager()
    hits = [
        [0,1,1,234],
        [0,3,10,34],
        [3,2,4,144],
    ]
    H.add_hits(hits)
    H.calc_pos()
