"""Code for conversion between FPGA channel numbers and geometry parameters in various schemes"""

import numpy as np
from pdb import set_trace as br


DTYPE_COOR = np.float32
COOR_ID = {'x': 0, 'y': 1, 'z': 2}


class Geometry:
    """Holds the geometry information and provides various conversion methods"""

    SL = {}

    def __init__(self, config):
        self.load_config(config)

    def load_config(self, config):
        """Loads geometry configuration"""
        self.NCHANNELS = config.NCHANNELS
        self.NLAYERS = config.NLAYERS
        self.NWIRES = int(config.NCHANNELS / config.NLAYERS)
        self.SL_FPGA_CHANNELS = config.SL_FPGA_CHANNELS
        self.SLS = tuple(self.SL_FPGA_CHANNELS.keys())
        self.XCELL = config.XCELL
        self.ZCELL = config.ZCELL
        self.TDRIFT = config.TDRIFT
        self.VDRIFT = self.XCELL*0.5 / self.TDRIFT
        self.GEO_CONDITIONS = config.GEO_CONDITIONS
        # Calculating local geometry properties
        self.WIRE_POSITIONS = self.wire_positions()
        self.CELL_BORDERS = self.cell_borders()
        self.SL_FRAME = self.sl_frame()


    def cell_borders(self):
        """Returns arrays of cell border positions for each layer"""
        borders = {}
        for l in range(1, self.NLAYERS+1):
            b = {}
            b['l'] = self.WIRE_POSITIONS[l]['x'] - 0.5 * self.XCELL
            b['r'] = self.WIRE_POSITIONS[l]['x'] + 0.5 * self.XCELL
            b['b'] = self.WIRE_POSITIONS[l]['z'] - 0.5*self.ZCELL
            b['t'] = self.WIRE_POSITIONS[l]['z'] + 0.5*self.ZCELL
            borders[l] = b
        return borders


    def wire_positions(self):
        """Returns arrays of x and y positions of all wires for each layer"""
        wires = np.arange(1, self.NWIRES+1, dtype=np.int8)
        pos = {}
        for l in range(1, self.NLAYERS+1):
            p = {}
            p['x'] = (wires - self.GEO_CONDITIONS['origin_wire']).astype(DTYPE_COOR) * self.XCELL + self.GEO_CONDITIONS['layer_shiftx'][l-1]
            p['z'] = np.array([self.GEO_CONDITIONS['layer_posz'][l-1]]*self.NWIRES, dtype=DTYPE_COOR)
            pos[l] = p
        return pos

    def sl_frame(self):
        """Returns local coordinates of a superlayer"""
        pos = {
            'l': np.min([b['l'] for b in self.CELL_BORDERS.values()]),
            'r': np.max([b['r'] for b in self.CELL_BORDERS.values()]),
            'b': np.min([b['b'] for b in self.CELL_BORDERS.values()]),
            't': np.max([b['t'] for b in self.CELL_BORDERS.values()]),
        }
        return pos



    def fill_hits_geometry(self, hits):
        """Fills the dataframe of input hits with calculated geometry information"""
        # Setting superlayer numbers
        hits_chfpga = []
        for sl, (fpga, ch_range) in self.SL_FPGA_CHANNELS.items():
            hits_chfpga.append((hits['FPGA'] == fpga) & (hits['TDC_CHANNEL'] >= ch_range[0]) & (hits['TDC_CHANNEL'] <= ch_range[1]))
        hits['SL'] = np.select(hits_chfpga, [0, 1, 2, 3], default=-1).astype(np.int8)
        # Setting layer numbers
        hits_chrem  = [
            (hits['TDC_CHANNEL'] % self.NLAYERS == 1 ),
            (hits['TDC_CHANNEL'] % self.NLAYERS == 2 ),
            (hits['TDC_CHANNEL'] % self.NLAYERS == 3 ),
            (hits['TDC_CHANNEL'] % self.NLAYERS == 0 ),
        ]
        hits['LAYER'] = np.select(hits_chrem, self.GEO_CONDITIONS['chrem_layer'], default=0).astype(np.uint8)
        # Setting normalized channel number inside chamber
        hits['TDC_CHANNEL_NORM'] = (hits['TDC_CHANNEL'] - self.NCHANNELS * (hits['SL']%2)).astype(np.uint8)
        # Setting wire number within the layer
        sel_phys = hits['SL'] >= 0
        hits['WIRE_NUM'] = ((hits['TDC_CHANNEL_NORM'] - 1) / 4 + 1).astype(np.uint8)
        # Setting wire positions
        hits_layer = [
            (hits['LAYER'] == 1),
            (hits['LAYER'] == 2),
            (hits['LAYER'] == 3),
            (hits['LAYER'] == 4),
        ]
        hits['Z_POS_WIRE'] = np.select(hits_layer, self.GEO_CONDITIONS['layer_posz'], default=0).astype(DTYPE_COOR)
        hits['X_POS_WIRE'] = (hits['WIRE_NUM'].astype(np.int8) - self.GEO_CONDITIONS['origin_wire']).astype(DTYPE_COOR) * self.XCELL
        hits['X_POS_WIRE'] = hits['X_POS_WIRE'] + np.select(hits_layer, self.GEO_CONDITIONS['layer_shiftx'], default=0).astype(DTYPE_COOR)
        hits.loc[~sel_phys, ['TDC_CHANNEL_NORM', 'WIRE_NUM', 'LAYER', 'X_POS_WIRE', 'Z_POS_WIRE']] = [0, 0, 0, 0.0, 0.0]


    def fill_hits_position(self, hits, reco=False):
        """Fills the dataframe of input hits with calculated position values"""
        if not reco:
            # Input hits are from the raw data
            hits['Z_POS'] = hits['Z_POS_WIRE']
            hits['X_POS_LEFT']  = hits['X_POS_WIRE'] - np.maximum(hits['TIMENS'], 0)*self.VDRIFT
            hits['X_POS_RIGHT']  = hits['X_POS_WIRE'] + np.maximum(hits['TIMENS'], 0)*self.VDRIFT
            sel_phys = hits['SL'] >= 0
            hits.loc[~sel_phys, ['X_POS_LEFT', 'X_POS_RIGHT']] = [0.0, 0.0]
        else:
            # Input hits are at the reconstruction level
            nhits = len(hits)
            hits_layer = [
                (hits['layer'] == 1),
                (hits['layer'] == 2),
                (hits['layer'] == 3),
                (hits['layer'] == 4),
            ]
            hits['posz'] = np.select(hits_layer, self.GEO_CONDITIONS['layer_posz'], default=0).astype(DTYPE_COOR)
            hits['posy'] = np.zeros(nhits, dtype=DTYPE_COOR)
            posx_wire = (hits['wire'].astype(np.int8) - self.GEO_CONDITIONS['origin_wire']).astype(DTYPE_COOR) * self.XCELL
            posx_wire = posx_wire + np.select(hits_layer, self.GEO_CONDITIONS['layer_shiftx'], default=0).astype(DTYPE_COOR)
            hits['lposx']  = (posx_wire - np.maximum(hits['time'], 0)*self.VDRIFT).astype(DTYPE_COOR)
            hits['rposx']  = (posx_wire + np.maximum(hits['time'], 0)*self.VDRIFT).astype(DTYPE_COOR)



