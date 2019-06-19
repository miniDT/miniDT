"""Configuration for the analysis"""

import numpy as np

### NCHANNELS = max n. channels to be considered to evaluate triplet patterns
NCHANNELS = 64                       # numbers of channels per chamber (one FPGA maps 2 chambers --> 128 channels per FPGA --> 2SLs)
### Virtual (FPGA, TDC_CHANNEL) pairs containing event/trigger information
EVENT_NR_CHANNELS = [(1,138), (1,137), (0,138), (0,137)]
CHANNELS_VIRTUAL = [(1, 129), (0, 139), (1, 139)]  # virtual channels that have to be stored
CHANNELS_TRIGGER = [(1, 129)]  # external trigger
# CHANNELS_TRIGGER = [(0, 139), (1, 139)]  # internal trigger (using OR of all the channels)
### Cell parameters
XCELL     = 42.                      # cell width in mm
ZCELL     = 13.                      # cell height in mm
ZCHAMB    = 550.                     # spacing betweeen chambers in mm
DURATION = {                         # duration in ns of different periods
    'orbit:bx': 3564,
    'orbit': 3564*25,
    'bx': 25.,
    'tdc': 25./30
}
TDRIFT    = 15.6*DURATION['bx']    # drift time in ns
VDRIFT    = XCELL*0.5 / TDRIFT     # drift velocity in mm/ns 
TRIGGER_TIME_ARRAY = np.array([DURATION['orbit'], DURATION['bx'], DURATION['tdc']])
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
TIME_OFFSET_SL = [0, 0, 0, 0]        # relative signal offset of each chamber
# TIME_OFFSET_SL = [111, 111, 106, 106]        # relative signal offset of each chamber
