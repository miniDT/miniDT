"""Configuration for the reconstruction code"""

import numpy as np
from math import pi

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0,      0,      0),
    1: (0,      0,      0),
    2: (100,    0,     50),
    3: (100,   50,      0)
}

# Rotation of each SL around the (X,Y,Z) axis
SL_ROTATION = {
    0: (0,      0,      0),
    1: (0,      0,      0.5*pi),
    2: (0,      0,      0),
    3: (0,      0,      0.5*pi)
}

# Precisions
DTYPE_COOR = np.float64
