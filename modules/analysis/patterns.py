"""CELL PATTERNS FOR MEANTIMER PATTERN MATCHING
patterns  =  dictionary of patterns (list of triplets of hits time in bx relative to orbit: BX + TDC/30), arranged by key being a string identifier used to select the proper mean-timing eq to be solved 
"""

import math
import numpy as np
import pandas as pd
from .config import NCHANNELS, TDRIFT, VDRIFT, ZCELL, MEANTIMER_CLUSTER_SIZE, MEANTIMER_SL_MULT_MIN
from pdb import set_trace as br

# Difference between t0 candidates that should be clustered together for the mean [ns]
MEAN_TZERO_DIFF = 4

############################################# MEANTIMER EQUATIONS
def meantimereq(pattern, timelist):
    """Function returning the expected t0 out of hits triples. None by default 
    tkey is the identifier of the univoque equation to be used given the pattern of hits in the triplet
    timelist is a len=3 list of hits time
    """
    patt = pattern[:-1]
    if patt in ('ABC','BCD'): 
        tzero = 0.25 * ( timelist[0] + 2*timelist[1] + timelist[2] - 2*TDRIFT)
        angle = math.atan(0.5 * (timelist[0] - timelist[2]) * VDRIFT / ZCELL)
    elif patt == 'ABD':
        tzero = 0.25 * ( 3*timelist[1] + 2*timelist[0] - timelist[2] - 2*TDRIFT)
        angle = math.atan(0.5 * (timelist[2] - timelist[1]) * VDRIFT / ZCELL)
    elif patt == 'ACD':
        tzero = 0.25 * ( 3*timelist[1] + 2*timelist[2] - timelist[0] - 2*TDRIFT)
        angle = math.atan(0.5 * (timelist[0] - timelist[1]) * VDRIFT / ZCELL)
    else:
        return None
    # Inverting angle if this is a left-side pattern
    if pattern[-1] == 'l':
        angle *= -1
    return tzero, angle


def tzero_clusters(tzeros):
    """Groups different tzero candidates into clusters"""
    clusters = []
    # Creating a dataframe with the data
    vals = []
    for sl, tz in tzeros.items():
        for tzero in tz:
            vals.append({'t0': tzero, 'sl': sl})
    if not vals:
        return None
    df = pd.DataFrame.from_dict(vals)
    df.sort_values('t0', inplace=True)
    # Adding a cluster column
    gr = df.sort_values('t0')['t0'].diff().fillna(0)
    gr[gr <= MEAN_TZERO_DIFF] = 0
    gr[gr > MEAN_TZERO_DIFF] = 1
    gr = gr.cumsum().astype(np.int8)
    df['cluster'] = gr
    # Counting number of layers and solutions in each cluster
    clusters = df.groupby('cluster')
    nSols = clusters.agg('size')
    nLayers = clusters['sl'].agg('nunique')
    clusters_valid = nSols[(nLayers >= MEANTIMER_SL_MULT_MIN) & (nSols >= MEANTIMER_CLUSTER_SIZE)].index
    if len(clusters_valid) < 1:
        return None
    return df[df['cluster'].isin(clusters_valid)]


def mean_tzero(tzeros):
    """Calculates the most probable t0 from multiple candidates of different meantimer solutions"""
    df = tzero_clusters(tzeros)
    if df is None:
        return -1, [], 0
    # Selecting the largest cluster
    gr = df.groupby('cluster')
    nSols = gr.agg('size')
    nSLs = gr['sl'].agg('nunique')
    cluster_id = nSols.idxmax()
    cluster = df.loc[df['cluster'] == cluster_id, 't0'].values
    return cluster.mean(), cluster, nSLs[cluster_id]


############################################# POSSIBLE HIT PATTERNS
PATTERNS = {}
### 3 ABC RIGHT
PATTERNS['ABCr']  = [ (1+x, 3+x,  2+x) for x in range(0, NCHANNELS, 4) ]
#A |1   x    |5   o    |9   o    |
#B     |3   x    |7   o    |
#C |2   x    |6   o    |10  o    |
#D     |4   o    |8   o    |
### 3 ABC LEFT
PATTERNS['ABCl'] = [ (5+x, 3+x,  6+x) for x in range(0, NCHANNELS, 4)[:-1] ]
#A |1   o    |5   x    |9   o    |
#B     |3   x    |7   o    |
#C |2   o    |6   x    |10  o    |
#D     |4   o    |8   o    |

### 3 BCD RIGHT
PATTERNS['BCDr']  = [ (3+x, 6+x,  4+x) for x in range(0, NCHANNELS, 4)[:-1] ]
#A |1   o    |5   o    |9   o    |
#B     |3   x    |7   o    |
#C |2   o    |6   x    |10  o    |
#D     |4   x    |8   o    |
### 3 BCD LEFT
PATTERNS['BCDl'] = [ (3+x, 2+x,  4+x) for x in range(0, NCHANNELS, 4) ]
#A |1   o    |5   o    |9   o    |
#B     |3   x    |7   o    |
#C |2   x    |6   o    |10  o    |
#D     |4   x    |8   o    |

### 3 ACD RIGHT
PATTERNS['ACDr']  = [ (1+x, 2+x,  4+x) for x in range(0, NCHANNELS, 4) ]
#A |1   x    |5   o    |9   o    |
#B     |3   o    |7   o    |
#C |2   x    |6   o    |10  o    |
#D     |4   x    |8   o    |
### 3 ACD LEFT
PATTERNS['ACDl'] = [ (5+x, 6+x,  4+x) for x in range(0, NCHANNELS, 4)[:-1] ]
#A |1   o    |5   x    |9   o    |
#B     |3   o    |7   o    |
#C |2   o    |6   x    |10  o    |
#D     |4   x    |8   o    |

### 3 ABD RIGHT
PATTERNS['ABDr']  = [ (1+x, 3+x,  4+x) for x in range(0, NCHANNELS, 4) ]
#A |1   x    |5   o    |9   o    |
#B     |3   x    |7   o    |
#C |2   o    |6   o    |10  o    |
#D     |4   x    |8   o    |
### 3 ABD LEFT
PATTERNS['ABDl'] = [ (5+x, 3+x,  4+x) for x in range(0, NCHANNELS, 4)[:-1] ]
#A |1   o    |5   x    |9   o    |
#B     |3   x    |7   o    |
#C |2   o    |6   o    |10  o    |
#D     |4   x    |8   o    |

# Transposed dictionary to quickly find pattern name
PATTERN_NAMES = {}
for name, patterns in PATTERNS.items():
    for pattern in patterns:
        PATTERN_NAMES[pattern] = name

# Lists of channels where hits from good events are expected AUGUST
ACCEPTANCE_CHANNELS = {
    0: range(1, NCHANNELS+1),
    1: range(1, NCHANNELS+1),
    2: range(1, NCHANNELS+1),
    3: range(1, NCHANNELS+1),
}
