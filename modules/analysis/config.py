"""Configuration for the hit analysis"""

import numpy as np

###### Number of channels in a single chamber
NCHANNELS = 64                      # one FPGA maps 2 chambers --> 128 channels per FPGA
NLAYERS = 4                         # number of layers in a single chamber
SL_FPGA_CHANNELS = {
    0: (0, (1, 64)),
    1: (0, (65, 128)),
    2: (1, (1, 64)),
    3: (1, (65, 128)),
}

###### Cell parameters
XCELL     = 42.        # cell width in mm
ZCELL     = 13.        # cell height in mm
ZCHAMB    = 550.       # spacing betweeen chambers in mm
DURATION  = {          # duration in ns of different periods
    'orbit:bx': 3564,
    'orbit': 3564*25,
    'bx': 25.,
    'tdc': 25./30
}
TDRIFT    = 15.6*DURATION['bx']    # drift time in ns
VDRIFT    = XCELL*0.5 / TDRIFT     # drift velocity in mm/ns 
TRIGGER_TIME_ARRAY = np.array([DURATION['orbit'], DURATION['bx'], DURATION['tdc']])

###### Description of geometry
GEO_CONDITIONS = {
    'chrem_layer': [1, 3, 2, 4,],
    'layer_shiftx': [-0.5*XCELL, 0.0, -0.5*XCELL, 0.0],
    'layer_posz': [1.5*ZCELL, 0.5*ZCELL, -0.5*ZCELL, -1.5*ZCELL],
    'origin_wire': 8  # in layer 2
}

###### Virtual (FPGA, TDC_CHANNEL) pairs containing trigger information
CHANNELS_VIRTUAL = [(1, 129), (0, 139), (1, 139)]  # virtual channels that have to be stored
CHANNELS_TRIGGER = [(1, 129)]  # external trigger
# CHANNELS_TRIGGER = [(0, 139), (1, 139)]  # internal trigger (using OR of all the channels)

### Minimum time [bx] between groups of hits with EVENT_NR to be considered as belonging to separate events
EVENT_TIME_GAP = 1000/DURATION['bx']
### Criteria for input hits for meantimer
NHITS_SL = (1, 10)  # (min, max) number of hits in a superlayer to be considered in the event
MEANTIMER_ANGLES = [(-0.2, 0.1), (-0.2, 0.1), (-0.1, 0.2), (-0.1, 0.2)]
MEANTIMER_CLUSTER_SIZE = 2  # minimum number of meantimer solutions in a cluster to calculate mean t0
MEANTIMER_SL_MULT_MIN = 2  # minimum number of different SLs in a cluster of meantimer solutions


# Parameters of the DAQ signals [must be optimised according to the exact setup performance]
TIME_OFFSET = 0                      # synchronization w.r.t external trigger
TIME_WINDOW = (-1000, 1500)             # time window (lower, higher) edge, after synchronization
# TIME_OFFSET_SL = [0, 0, 0, 0]        # relative signal offset of each chamber
# TIME_OFFSET_SL = [111, 111, 106, 106]        # relative signal offset of each chamber [RUN617]
TIME_OFFSET_SL = [127, 127, 123, 122]        # relative signal offset of each chamber
