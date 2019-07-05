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

    def fill_hits_dataframe(self, hits):
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
        hits['TDC_CHANNEL_NORM'] = -1
        hits.loc[hits['SL'] >= 0, 'TDC_CHANNEL_NORM'] = (hits['TDC_CHANNEL'] - self.NCHANNELS * (hits['SL']%2)).astype(np.uint8)
        # Setting wire positions
        hits['Z_POS_WIRE'] = (hits['LAYER'].astype(np.float16) - 0.5) * self.ZCELL
        hits['X_POS_WIRE'] = (((hits['TDC_CHANNEL'] - 1) / 4).astype(np.float16) + 0.5) * self.XCELL
        hits['X_POS_WIRE'] = hits['X_POS_WIRE'] + np.select(hits_chrem, self.GEO_CONDITIONS['chrem_shiftx'], default=0).astype(np.float16)

