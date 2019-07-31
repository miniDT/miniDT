"""Configuration for the reconstruction code"""

import numpy as np
from math import pi

sl_w = 700     # chamber width (along X)
sl_l = 700     # chamber length (along Y)
sl_h = 62       # chamber height
dsl = 782       # Z of chamber 2

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.5*sl_w +10,   0.5*sl_l,      0.5*sl_h),
    1: (0.5*sl_l,      0.5*sl_w +10,  1.5*sl_h),
    2: (0.5*sl_w,      0.5*sl_l,      0.5*sl_h + dsl),
    3: (0.5*sl_l,      0.5*sl_w,      1.5*sl_h + dsl)
}

# Rotation of each SL around the (X,Y,Z) axis
SL_ROTATION = {
    0: (0,        0,          0),
    1: (0,        0,     0.5*pi),
    2: (0,        0,          0),
    3: (0,        0,     0.5*pi)
}

# Selection of hits to use for the fits
TIMEBOX = (-5, 380)

# Precisions
DTYPE_COOR = np.float64
FIT_CHI2_MAX = 2.0

# Colors of fitted tracks
TRACK_COLORS = ['limegreen', 'forestgreen', 'goldenrod', 'peru', 'tomato', 'maroon']

# Fit parameters
NHITS_MIN_LOCAL = 3
