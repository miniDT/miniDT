"""Code for conversion between FPGA channel numbers and geometry parameters in various schemes"""

from pdb import set_trace as br
import numpy as np


class Geometry:
    """Holds the geometry information and provides various conversion methods"""

    def __init__(self, config):
        self.load_config(config)

    def load_config(self, config):
        """Loads geometry configuration"""
        self.NCHANNELS = config.NCHANNELS
        self.NLAYERS = config.NLAYERS
        self.SL_FPGA_CHANNELS = config.SL_FPGA_CHANNELS
        self.XCELL = config.XCELL
        self.ZCELL = config.ZCELL
        self.TDRIFT = config.TDRIFT
        self.VDRIFT = self.XCELL*0.5 / self.TDRIFT
        self.GEO_CONDITIONS = config.GEO_CONDITIONS

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
        hits['Z_POS_WIRE'] = np.select(hits_layer, self.GEO_CONDITIONS['layer_posz'], default=0).astype(np.float16)
        hits['X_POS_WIRE'] = (hits['WIRE_NUM'].astype(np.int8) - self.GEO_CONDITIONS['origin_wire']).astype(np.float16) * self.XCELL
        hits['X_POS_WIRE'] = hits['X_POS_WIRE'] + np.select(hits_layer, self.GEO_CONDITIONS['layer_shiftx'], default=0).astype(np.float16)
        hits.loc[~sel_phys, ['TDC_CHANNEL_NORM', 'WIRE_NUM', 'LAYER', 'X_POS_WIRE', 'Z_POS_WIRE']] = [0, 0, 0, 0.0, 0.0]

    def fill_hits_position(self, hits):
        """Fills the dataframe of input hits with calculated position values"""
        hits['Z_POS'] = hits['Z_POS_WIRE']
        hits['X_POS_LEFT']  = hits['X_POS_WIRE'] - np.maximum(hits['TIMENS'], 0)*self.VDRIFT
        hits['X_POS_RIGHT']  = hits['X_POS_WIRE'] + np.maximum(hits['TIMENS'], 0)*self.VDRIFT
        sel_phys = hits['SL'] >= 0
        hits.loc[~sel_phys, ['X_POS_LEFT', 'X_POS_RIGHT']] = [0.0, 0.0]


