#!/usr/bin/env python
from pdb import set_trace as br
from time import clock
from multiprocessing import Pool
import math
import numpy as np
import pandas as pd
import itertools
import os 
import sys
import operator

# Importing custom code snippets
from modules.analysis.patterns import PATTERNS, PATTERN_NAMES, ACCEPTANCE_CHANNELS, MEAN_TZERO_DIFF, meantimereq, mean_tzero, tzero_clusters
from modules.analysis.config import NCHANNELS, XCELL, ZCELL, TDRIFT, VDRIFT, CHANNELS_TRIGGER, CHANNELS_VIRTUAL, EVENT_NR_CHANNELS
from modules.analysis.config import EVENT_TIME_GAP, TIME_OFFSET, TIME_OFFSET_SL, TIME_WINDOW, DURATION, TRIGGER_TIME_ARRAY
from modules.analysis.config import NHITS_SL, MEANTIMER_ANGLES, MEANTIMER_CLUSTER_SIZE, MEANTIMER_SL_MULT_MIN
from modules.utils import print_progress



############################################# INPUT ARGUMENTS 
import argparse
parser = argparse.ArgumentParser(description='Offline analysis of unpacked data. t0 id performed based on pattern matching.')
parser.add_argument('inputs', metavar='FILE', help='Unpacked input file to analyze', nargs='+')
parser.add_argument('-a', '--accepted',  help='Save only events that passed acceptance cuts', action='store_true', default=False)
parser.add_argument('-c', '--csv',  help='Print final selected hits into CSV files', action='store_true', default=False)
parser.add_argument('--chambers',  help='Minimum number of chambers with 1+ hits', action='store', default=4, type=int)
parser.add_argument('-d', '--double_hits',  help='Accept only events with 2+ hits in a cell', action='store_true', default=False)
parser.add_argument('-e', '--event',  help='Split hits in events based on external trigger', action='store_true', default=False)
parser.add_argument('-E', '--events', metavar='N',  help='Only process events with specified numbers', type=int, default=None, nargs='+')
parser.add_argument('-g', '--group', metavar='N', type=int, help='Process input files sequentially in groups of N', action='store', default=999999)
parser.add_argument('-l', '--layer',   action='store', default=None, dest='layer',   type=int, help='Layer to process [default: process all 4 layers]')
parser.add_argument('-m', '--max_hits',   action='store', default=200, dest='max_hits',   type=int, help='Maximum number of hits allowed in one event [default: 200]')
parser.add_argument('-n', '--number', action='store', default=None,  dest='number', type=int, help='Number of hits to analyze. (Note: this is applied to each file if multiple files are analyzed with -g)')
parser.add_argument('-r', '--root',  help='Print output to a ROOT friendly text file', action='store_true', default=False)
parser.add_argument('-s', '--suffix',  action='store', default=None, help='Suffix to add to output file names', type=str)
parser.add_argument('-t', '--triplets',  help='Do triplet search', action='store_true', default=False)
parser.add_argument('-u', '--update_tzero',  help='Update TIME0 with meantimer solution', action='store_true', default=False)
parser.add_argument('-v', '--verbose',  help='Increase verbosity of the log', action='store', default=0)
args = parser.parse_args()
for file_path in args.inputs:
    if not os.path.exists(os.path.expandvars(file_path)):
        print('--- ERROR ---')
        print('  \''+args.input+'\' file not found')
        print('  please point to the correct path to the file containing the unpacked data' )
        print()
        exit()

VERBOSE = int(args.verbose)
EVT_COL = 'EVENT_NR' if args.event else 'ORBIT_CNT'

#                         / z-axis (beam direction)
#                        .
#                       .
#                  ///////////|   
#                 ///////////||
#                ||||||||||||||
#                ||||||||||||||  SL 1/3
#                ||||||||||||/
#                |||||||||||/
#  y-axis       .          .
#(vertical     .          .
# pointing    .          .
# upward)    .          .
#   ^       .          .           y-axis
#   |   ///////////|  .            ^ 
#   |  ///////////|| .             |  z-axis 
#   | ||||||||||||||.              | / 
#   | ||||||||||||||  SL 0/2       |/
#   | ||||||||||||/                +-------> x-axis 
#   | |||||||||||/                 
#   +-------------> x-axis (horizontal)


#    layer #             cell numbering scheme
#      1         |    1    |    5    |    9    |
#      2              |    3    |    7    |   11    |
#      3         |    2    |    6    |   10    |
#      4              |    4    |    8    |   12    |

def analyse_parallel(args):
    """Wrapper around the main function to properly pass arguments"""
    return analyse(*args)

############################################# ANALYSIS
def analyse(allhits, SL):

    idx = allhits['SL'] == SL

    # # Excluding groups that have multiple time measurements with the same channel
    # # They strongly degrade performance of meantimer
    # idx = dfhits.loc[dfhits['TDC_CHANNEL_NORM'] <= NCHANNELS].groupby(EVT_COL).filter(lambda x: x['TDC_CHANNEL_NORM'].size == x['TDC_CHANNEL_NORM'].nunique()).index

    # correct hits time for tzero
    allhits.loc[idx, 'TIMENS'] = allhits['TIME_ABS'] - allhits['TIME0']
    
    # Assign hits position (left/right wrt wire)
    allhits.loc[idx, 'X_POS_LEFT']  = ((allhits['TDC_CHANNEL_NORM']-0.5).floordiv(4) + allhits['X_POSSHIFT'])*XCELL + XCELL/2 - np.maximum(allhits['TIMENS'], 0)*VDRIFT
    allhits.loc[idx, 'X_POS_RIGHT'] = ((allhits['TDC_CHANNEL_NORM']-0.5).floordiv(4) + allhits['X_POSSHIFT'])*XCELL + XCELL/2 + np.maximum(allhits['TIMENS'], 0)*VDRIFT


def calc_event_numbers(allhits, runnum):
    """Calculates event number for groups of hits based on trigger hits"""
    # Creating a dataframe to be filled with hits from found events (for better performance)
    hits = allhits.loc[:1, ['EVENT_NR', 'TIME0']]
    # Selecting only hits containing information about the event number or trigger signals
    channels = CHANNELS_TRIGGER
    sel = pd.Series(False, allhits.index)
    for ch in channels:
        sel = sel | ((allhits['FPGA'] == ch[0]) & (allhits['TDC_CHANNEL'] == ch[1]))
    # Selecting hits that have to be grouped by time
    ev_hits = allhits.loc[sel]
    print('### Grouping hits by their time of arrival')
    # Creating the list of hits with 1 on jump in BX
    evt_group = (ev_hits['ORBIT_CNT'].astype(np.uint64)*DURATION['orbit:bx'] + ev_hits['BX_COUNTER']).sort_values().diff().fillna(0).astype(np.uint64)
    evt_group[evt_group <= EVENT_TIME_GAP] = 0
    evt_group[evt_group > EVENT_TIME_GAP] = 1
    # Calculating cumulative sum to create group ids
    evt_group = evt_group.cumsum()
    # Adding column to be used for grouping hits with event number and trigger
    allhits['EVENT_NR'] = evt_group
    allhits['EVENT_NR'] = allhits['EVENT_NR'].fillna(-1).astype(int)
    # Getting back rows with relevant channels with grouping column updated
    ev_hits = allhits.loc[sel]
    ev_hits.set_index(['FPGA', 'TDC_CHANNEL'], inplace=True)
    # Checking each group to calculate event number for it
    evt_groups = ev_hits.groupby('EVENT_NR')
    n_groups = len(evt_groups)
    n_groups_done = 0
    # Creating a dataframe with 1 row per event
    df_events = pd.DataFrame(data={'EVENT_ID': list(evt_groups.groups.keys())})
    df_events['TIME0'] = -1
    df_events['EVENT_NR'] = -1
    df_events['TIMEDIFF_TRG_20'] = -1e9
    df_events['TIMEDIFF_TRG_21'] = -1e9
    df_events.set_index('EVENT_ID', inplace=True)
    df_events.sort_index(inplace=True)
    # Calculating event number for each group of hits
    for grp, df in evt_groups:
        print_progress(n_groups_done, n_groups)
        n_groups_done += 1
        df = df.sort_index()
        try:
            vals_int = df['TDC_MEAS'].reindex(EVENT_NR_CHANNELS, fill_value=0)
        except Exception:
            # Removing duplicate entries with the same channel value (very rare occasion)
            if VERBOSE:
                print('WARNING: duplicate entries with the same channel for event number:')
                print(df[['ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS']])
            df = df[~df.index.duplicated(keep='first')]
            vals_int = df['TDC_MEAS'].reindex(EVENT_NR_CHANNELS, fill_value=0)

        evt_id = grp
        # Skipping if only one specific event should be processed
        if args.events and evt_id not in args.events:
            continue

        df_events.loc[grp, 'EVENT_NR'] = evt_id

        # Getting time and orbit number of the event after duplicates were eliminated
        time_event, orbit_event = None, None
        for ch in CHANNELS_TRIGGER:
            if ch in df.index:
                time_event, orbit_event = df.loc[ch, ('TIME_ABS', 'ORBIT_CNT')]
                break

        # Looking for other hits within the time window of the event, taking into account latency
        # set ttrig based on run number (HARDCODED)
        time_offset = TIME_OFFSET
        # Defining t0 as time of the trigger channel corrected by latency offset
        tzero = time_event + time_offset
        ############################################
        # # FIXME: Correcting TIME0 for this particular event
        # tzero += 4.0
        ############################################
        event_window = (tzero + TIME_WINDOW[0], tzero + TIME_WINDOW[1])

        try:
            window = allhits['TIME_ABS'].between(event_window[0], event_window[1], inclusive=False)
            # print('event: {0:d}  duration: {1:.3f}'.format(evt_id, clock() - start))
        except Exception as e:
            print('WARNING: Exception when calculating window')

        df_events.loc[grp, ['EVENT_NR', 'TIME0']] = (evt_id, tzero)

        # Storing hits of the event with corresponding event number and t0
        idx = allhits.index[window | (allhits['EVENT_NR'] == grp)]
        hits = hits.append(pd.DataFrame(np.array([evt_id, tzero]*len(idx)).reshape([-1, 2]), index=idx, columns=['EVENT_NR', 'TIME0']))
    # Updating hits in the main dataframe with EVENT_NR and TIME0 values from detected events
    hits['EVENT_NR'] = hits['EVENT_NR'].astype(int)
    allhits.loc[hits.index, ['EVENT_NR', 'TIME0']] = hits[['EVENT_NR', 'TIME0']]

    # Creating a column with time passed since last event
    df_events.set_index('EVENT_NR', inplace=True)
    # Removing events that have no hits
    if -1 in df_events.index:
        df_events.drop(-1, inplace=True)
    return df_events

def meantimer_results(df_hits, verbose=False):
    """Run meantimer over the group of hits"""
    sl = df_hits['SL'].iloc[0]
    # Getting a TIME column as a Series with TDC_CHANNEL_NORM as index
    df_time = df_hits.loc[:, ['TDC_CHANNEL_NORM', 'TIME_ABS', 'LAYER']]
    df_time.sort_values('TIME_ABS', inplace=True)
    # Split hits in groups where time difference is larger than maximum event duration
    grp = df_time['TIME_ABS'].diff().fillna(0)
    event_width_max = 1.1*TDRIFT
    grp[grp <= event_width_max] = 0
    grp[grp > 0] = 1
    grp = grp.cumsum().astype(np.uint16)
    df_time['grp'] = grp
    # Removing groups with less than 3 unique hits
    df_time = df_time[df_time.groupby('grp')['TDC_CHANNEL_NORM'].transform('nunique') >= 3]
    # Determining the TIME0 using triplets [no external trigger]
    tzeros = []
    angles = []
    # Processing each group of hits
    patterns = PATTERN_NAMES.keys()
    for grp, df_grp in df_time.groupby('grp'):
        df_grp.set_index('TDC_CHANNEL_NORM', inplace=True)
        # Selecting only triplets present among physically meaningful hit patterns
        channels = set(df_grp.index.astype(np.int16))
        triplets = set(itertools.permutations(channels, 3))
        triplets = triplets.intersection(patterns)
        # Grouping hits by the channel for quick triplet retrieval
        times = df_grp.groupby(df_grp.index)['TIME_ABS']
        # Analysing each triplet
        for triplet in triplets:
            triplet_times = [times.get_group(ch).values for ch in triplet]
            for t1 in triplet_times[0]:
                for t2 in triplet_times[1]:
                    for t3 in triplet_times[2]:
                        timetriplet = (t1, t2, t3)
                        if max(timetriplet) - min(timetriplet) > 1.1*TDRIFT:
                            continue
                        pattern = PATTERN_NAMES[triplet]
                        mean_time, angle = meantimereq(pattern, timetriplet)
                        if verbose:
                            print('{4:d} {0:s}: {1:.0f}  {2:+.2f}  {3}'.format(pattern, mean_time, angle, triplet, sl))
                        # print(triplet, pattern, mean_time, angle)
                        if not MEANTIMER_ANGLES[sl][0] < angle < MEANTIMER_ANGLES[sl][1]:
                            continue
                        tzeros.append(mean_time)
                        angles.append(angle)

    return tzeros, angles


def save_root(df_all, df_events, output_path):
    """Prints output to a text file with one event per line, sequence of hits in a line"""
    if df_all is None or df_all.empty:
        print('WARNING: No hits for writing into a text file')
        return
    # Selecting only physical or trigger hits [for writing empty events as well]
    sel = df_all['TIME0'] > 0
    for ch in CHANNELS_TRIGGER:
        sel = sel | ((df_all['FPGA'] == ch[0]) & (df_all['TDC_CHANNEL'] == ch[1]))
    df_all = df_all[sel]
    events = df_all.groupby('EVENT_NR')
    layers = range(4)
    print('### Writing {0:d} hits in {1:d} events to file: {2:s}'.format(df_all.shape[0], len(events), output_path))
    with open(output_path, 'w') as outfile:
        for event, df in events:
            ch_sel = (df['TIME0'] > 0)
            n_layer_hits = df.loc[ch_sel].sort_values('SL').groupby('SL').size().reindex(layers).fillna(0).astype(int).tolist()
            nhits = df.loc[ch_sel].shape[0]
            # if nhits < 3:
            #     continue
            orbit, tzero = df.iloc[0][['ORBIT_CNT', 'TIME0']]
            # tzero_meantimer, meantimer_mult, meantimer_min, meantimer_max = df_events.loc[event, ['MEANTIMER_MEAN', 'MEANTIMER_MULT', 'MEANTIMER_MIN', 'MEANTIMER_MAX']]
            # Merging all hits
            line = '{0:d} {1:d}'.format(event, nhits) 
            # line = '{0:d} {1:d} {2:d} {3} {4:d}'.format(event, int(orbit), nhits, n_layer_hits, int(tzero)) 
            # line = '{0:d} {1:d} {2:d} {3} {4:d} {5:.1f} {6:d} {7:d}'.format(event, int(orbit), nhits, n_layer_hits, int(tzero), 
                                                                            # tzero_meantimer-tzero, int(meantimer_max - meantimer_min), int(meantimer_mult))
            if nhits > 0:
                line += ' ' + ' '.join(['{0:.0f} {1:.0f} {2:.3e} {3:.3e} {4:.1f}'.format(*values)
                                       for values in df.loc[ch_sel, ['SL', 'LAYER', 'X_POS_LEFT', 'X_POS_RIGHT', 'TIMENS']].values])
            outfile.write(line+'\n')


############################################# READING DATA FROM CSV INPUT
def read_data(input_files, runnum):
    """
    Reading data from CSV file into a Pandas dataframe, applying selection and sorting
    """
    # Reading each file and merging into 1 dataframe
    hits = []
    for index, file in enumerate(input_files):
        skipLines = 0
        # Skipping the old buffer content of the boards that is dumped at the beginning of the run
        if 'data_000000' in file:
            skipLines = range(1,131072)
        df = pd.read_csv(file, nrows=args.number, skiprows=skipLines, engine='c')
        # Removing possible incomplete rows e.g. last line of last file
        df.dropna(inplace=True)
        # Converting to memory-optimised data types
        for name in ['HEAD', 'FPGA', 'TDC_CHANNEL', 'TDC_MEAS']:
            df[name] = df[name].astype(np.uint8)
        for name in ['BX_COUNTER']:
            df[name] = df[name].astype(np.uint16)
        for name in ['ORBIT_CNT']:
            df[name] = df[name].astype(np.uint32)
        hits.append(df)
    allhits = pd.concat(hits, ignore_index=True, copy=False)
    df_events = None
    print('### Read {0:d} hits from {1:d} input files'.format(allhits.shape[0], len(hits)))
    # retain all words with HEAD=1
    allhits.drop(allhits.index[allhits['HEAD'] != 1], inplace=True)
    # Removing unused columns to save memory foot-print
    allhits.drop('HEAD', axis=1, inplace=True)
    ### # Increase output of all channels with id below 130 by 1 ns --> NOT NEEDED
    ### allhits.loc[allhits['TDC_CHANNEL'] <= 130, 'TDC_MEAS'] = allhits['TDC_MEAS']+1 
    # Calculate absolute time in ns of each hit
    allhits['TIME_ABS'] = (allhits['ORBIT_CNT'].astype(np.float64)*DURATION['orbit'] + 
                           allhits['BX_COUNTER'].astype(np.float64)*DURATION['bx'] + 
                           allhits['TDC_MEAS'].astype(np.float64)*DURATION['tdc']).astype(np.float64)
    # Adding columns to be calculated
    nHits = allhits.shape[0]
    allhits['TIME0'] = np.zeros(nHits, dtype=np.float64)
    allhits['EVENT_NR'] = np.ones(nHits, dtype=np.uint32) * -1
    # Calculating additional info about the hits
    conditions  = [
        (allhits['TDC_CHANNEL'] % 4 == 1 ),
        (allhits['TDC_CHANNEL'] % 4 == 2 ),
        (allhits['TDC_CHANNEL'] % 4 == 3 ),
        (allhits['TDC_CHANNEL'] % 4 == 0 ),
    ]
    chanshift_x = [  0,            -1,           0,            -1,        ]
    layer_z     = [  1,            3,            2,            4,         ]
    pos_z       = [  ZCELL*3.5,    ZCELL*1.5,    ZCELL*2.5,    ZCELL*0.5, ]
    posshift_x  = [  0,            0,            0.5,          0.5,       ]
    # Adding columns
    allhits['LAYER']      = np.select(conditions, layer_z,      default=0).astype(np.uint8)
    allhits['X_CHSHIFT']  = np.select(conditions, chanshift_x,  default=0).astype(np.int8)
    allhits['X_POSSHIFT'] = np.select(conditions, posshift_x,   default=0).astype(np.float16)
    allhits['Z_POS']      = np.select(conditions, pos_z,        default=0).astype(np.float16)

    # conditions  = FPGA number and TDC_CHANNEL in range
    # SL         <- superlayer = chamber number from 0 to 3 (0,1 FPGA#0 --- 2,3 FPGA#1)
    conditions_SL = [
        ((allhits['FPGA'] == 0) & (allhits['TDC_CHANNEL'] <= NCHANNELS )),
        ((allhits['FPGA'] == 0) & (allhits['TDC_CHANNEL'] > NCHANNELS ) & (allhits['TDC_CHANNEL'] <= 2*NCHANNELS )),
        ((allhits['FPGA'] == 1) & (allhits['TDC_CHANNEL'] <= NCHANNELS )),
        ((allhits['FPGA'] == 1) & (allhits['TDC_CHANNEL'] > NCHANNELS ) & (allhits['TDC_CHANNEL'] <= 2*NCHANNELS )),
    ]
    allhits['SL'] = np.select(conditions_SL, [0, 1, 2, 3], default=-1).astype(np.int8)
    # Correcting absolute time by per-chamber latency
    for sl in range(4):
        sel = allhits['SL'] == sl
        allhits.loc[sel, 'TIME_ABS'] = allhits.loc[sel, 'TIME_ABS'] + TIME_OFFSET_SL[sl]
    
    # # Removing hits from unused layers
    # if args.layer is not None:
    #     allhits.drop(allhits[~allhits['SL'].isin([args.layer, -1])].index, inplace=True)

    # define channel within SL
    allhits['TDC_CHANNEL_NORM'] = (allhits['TDC_CHANNEL'] - NCHANNELS * (allhits['SL']%2)).astype(np.uint8)

    # Detecting events based on trigger signals
    if args.event:
        df_events = calc_event_numbers(allhits, runnum)
    else:
        # Grouping hits separated by large time gaps together
        allhits.sort_values('TIME_ABS', inplace=True)
        grp = allhits['TIME_ABS'].diff().fillna(0)
        grp[grp <= 1.1*TDRIFT] = 0
        grp[grp > 0] = 1
        grp = grp.cumsum().astype(np.int32)
        allhits['EVENT_NR'] = grp
        events = allhits.groupby('EVENT_NR')
        nHits = events.size()
        nHits_unique = events['TDC_CHANNEL'].nunique()
        nSL = events['SL'].nunique()
        # Selecting only events with manageable numbers of hits
        events = nHits.index[(nSL >= args.chambers) & (nHits_unique >= (MEANTIMER_CLUSTER_SIZE * 3)) & (nHits <= args.max_hits)]
        # Marking events that don't pass the basic selection
        sel = allhits['EVENT_NR'].isin(events)
        allhits.loc[~sel, 'EVENT_NR'] = -1
        df_events = pd.DataFrame(data={'EVENT_NR': events})
        df_events['TIME0'] = -1
        df_events['TRG_BITS'] = -1
        df_events['TIMEDIFF_TRG_20'] = -1e9
        df_events['TIMEDIFF_TRG_21'] = -1e9
        df_events.set_index('EVENT_NR', inplace=True)
    # Removing hits with no event number
    allhits.drop(allhits.index[allhits['EVENT_NR'] == -1], inplace=True)
    # Calculating event times
    df_events['TIME0_BEFORE'] = df_events['TIME0'].diff().fillna(0)
    df_events['TIME0_AFTER'] = df_events['TIME0'].diff(-1).fillna(0)
    
    # Removing hits with irrelevant tdc channels
    sel = allhits['TDC_CHANNEL_NORM'] <= NCHANNELS
    for ch in CHANNELS_VIRTUAL:
        sel = sel | ((allhits['FPGA'] == ch[0]) & (allhits['TDC_CHANNEL'] == ch[1]))
    allhits.drop(allhits.index[~sel], inplace=True)
    # Removing events that don't pass acceptance cuts
    if args.accepted:
        select_accepted_events(allhits, df_events)

    nHits = allhits.shape[0]
    # Adding extra columns to be filled in the analyse method
    allhits['TIMENS'] = np.zeros(nHits, dtype=np.float16)
    allhits['X_POS_LEFT'] = np.zeros(nHits, dtype=np.float32)
    allhits['X_POS_RIGHT'] = np.zeros(nHits, dtype=np.float32)


    #############################################
    ### DATA HANDLING 

    if VERBOSE:
        print('')
        print('dataframe size                   :', len(allhits))
        print('')

    if VERBOSE:
        print('dataframe size (no trigger hits) :', len(allhits))
        print('')
        print('min values in dataframe')
        print(allhits[['TDC_CHANNEL','TDC_CHANNEL_NORM','TDC_MEAS','BX_COUNTER','ORBIT_CNT']].min())
        print('')
        print('max values in dataframe')
        print(allhits[['TDC_CHANNEL','TDC_CHANNEL_NORM','TDC_MEAS','BX_COUNTER','ORBIT_CNT']].max())
        print('')

    return allhits, df_events


def event_accepted(df, cut_max_hits=False):
    """Checks whether the event passes acceptance cuts"""
    # Calculating numbers of hit layers in each chamber
    nLayers = df.groupby('SL')['LAYER'].agg('nunique')
    # Skipping events that have no minimum number of chambers with 3+ layers of hits
    if nLayers[nLayers >= 3].shape[0] < MEANTIMER_SL_MULT_MIN:
        return False
    # Calculating numbers of hits in each chamber
    grp = df.groupby('SL')['TDC_CHANNEL_NORM']
    nHits = grp.agg('nunique')
    # Skipping if has at least one chamber with too few hits
    if nHits[nHits >= NHITS_SL[0]].shape[0] < args.chambers:
        return False
    nHits = grp.agg('size')
    # Skipping if has at least one chamber with too many hits
    if cut_max_hits and nHits[nHits > NHITS_SL[1]].shape[0] > 0:
        return False
    # return True
    # Skipping events that don't have the minimum number of similar meantimer solutions
    tzeros_all = {}
    # Starting from SLs with smallest N of hits
    sl_ids = nHits.loc[nLayers >= 3].sort_values().index
    nSLs = len(sl_ids)
    nSLs_meant = 0
    event = df.iloc[0]['EVENT_NR']
    for iSL, SL in enumerate(sl_ids):
        # Stopping if required number of SLs can't be regardless of the following SLs
        # if nSLs - iSL + nSLs_meant < MEANTIMER_SL_MULT_MIN:
        #     break
        tzeros = meantimer_results(df[df['SL'] == SL])[0]
        if len(tzeros) > 0:
            nSLs_meant += 1
        tzeros_all[SL] = tzeros
    tzero, tzeros, nSLs = mean_tzero(tzeros_all)
    if len(tzeros) < MEANTIMER_CLUSTER_SIZE:
        return False
    return tzero, tzeros, tzeros_all


def select_accepted_events(allhits, events):
    """Removes events that don't pass acceptance cuts"""
    print('### Removing events outside acceptance')
    hits = allhits[allhits['TDC_CHANNEL_NORM'] <= NCHANNELS]
    sel = pd.concat([(hits['SL'] == sl) & (hits['TDC_CHANNEL_NORM'].isin(ch)) 
                    for sl, ch in ACCEPTANCE_CHANNELS.items()], axis=1).any(axis=1)
    groups = hits[sel].groupby('EVENT_NR')
    events_accepted = []
    n_events = len(groups)
    n_events_processed = 0
    print('### Checking {0:d} events'.format(n_events))
    events['CELL_HITS_MULT_MAX'] = 1
    events['CELL_HITS_DT_MIN'] = -1
    events['CELL_HITS_DT_MAX'] = -1
    # sl_channels = {2: [3,4,5,6], 3: [1,2,3,4]}
    # sl_channels = {2: [5,6,7,8], 3: [3,4,5,6]}
    # sl_channels = {2: [7,8,9,10], 3: [5,6,7,8]}
    sl_channels = None
    for event, df in groups:
        n_events_processed += 1
        print_progress(n_events_processed, n_events)
        # Accepting only specified events if provided
        if args.events and event not in args.events:
            continue
        # Selecting only events that have 1+ hits in a single cell
        if args.double_hits:
            cellHits = df.groupby(['SL', 'TDC_CHANNEL_NORM'])['TIME_ABS']
            nHits_max = cellHits.agg('size').max()
            if nHits_max < 2:
                continue
            dt_min = 999
            dt_max = -1
            for cell, hits in cellHits:
                if len(hits) < 2:
                    continue
                hits_sorted = np.sort(hits.values)
                dt_min = min(dt_min, hits_sorted[1] - hits_sorted[0])
                dt_max = max(dt_max, hits_sorted[-1] - hits_sorted[0])
            events.loc[event, ['CELL_HITS_MULT_MAX', 'CELL_HITS_DT_MIN', 'CELL_HITS_DT_MAX']] = [nHits_max, dt_min, dt_max]
            # Accepting the event
            events_accepted.append(event)
        # Skipping events that don't have hits exactly in the defined channels
        channels_ok = True
        if sl_channels:
            for sl, chs in sl_channels.items():
                if sorted(df[df['SL']==sl]['TDC_CHANNEL_NORM'].tolist()) == chs:
                    continue
                channels_ok = False
                break
            if not channels_ok:
                continue
        tzero_result = event_accepted(df, cut_max_hits=args.event)
        if not tzero_result:
            continue
        # Checking event for acceptance with different t0 candidates
        tzero, tzeros, tzeros_all = tzero_result
        df_clusters = tzero_clusters(tzeros_all)
        nClusters = df_clusters['cluster'].nunique()
        # Skipping event if no t0 cluster found
        if nClusters < 1:
            continue
        tzero = -1.0
        for cluster_id, df_cluster in df_clusters.groupby('cluster'):
            if len(df_cluster) < MEANTIMER_CLUSTER_SIZE:
                continue
            cluster = df_cluster['t0'].values
            tzero_cand = cluster.mean()
            # Trying acceptance cuts with a subset of hits from this orbit within the event time window
            window = ((df['TIME_ABS'] >= (tzero_cand + TIME_WINDOW[0])) & (df['TIME_ABS'] <= (tzero_cand + TIME_WINDOW[1]))).index
            tzero_result_cand = event_accepted(df.loc[window], cut_max_hits=True)
            if not tzero_result_cand:
                continue
            tzero = tzero_cand
            break
        if tzero < 0:
            continue
        # Accepting the event
        events_accepted.append(event)
        # Updating the TIME0 with meantimer result
        if args.update_tzero or not args.event:
            events.loc[event, 'TIME0'] = tzero
            if args.event:
                # Updating t0 of all hits directly if using external trigger
                allhits.loc[allhits['EVENT_NR'] == event, 'TIME0'] = tzero
            else:
                # Removing the old event number and applying only to hits in the window
                allhits.loc[allhits['EVENT_NR'] == event, 'EVENT_NR'] = -1
                # Updating t0 of hits in the event time window
                start = tzero + TIME_WINDOW[0]
                end = tzero + TIME_WINDOW[1]
                window = (allhits['TIME_ABS'] >= start) & (allhits['TIME_ABS'] <= end)
                allhits.loc[window, ['TIME0', 'EVENT_NR']] = [tzero, event]
    events.drop(events.index[~events.index.isin(events_accepted)], inplace=True)
    allhits.drop(allhits.index[~allhits['EVENT_NR'].isin(events_accepted)], inplace=True)
    print('### Selected {0:d}/{1:d} events in acceptance'.format(len(events_accepted), n_events))


def sync_triplets(results, df_events):
    """Synchronise events from triplet results in different SLs that were processed in parallel"""
    df_events['MEANTIMER_SL_MULT'] = -1
    df_events['MEANTIMER_MIN'] = -1
    df_events['MEANTIMER_MAX'] = -1
    df_events['MEANTIMER_MEAN'] = -1
    df_events['MEANTIMER_MULT'] = -1
    df_events['HITS_MULT'] = -1
    df_events['HITS_MULT_ACCEPTED'] = -1
    if not results:
        return
    groups = pd.concat([result[2] for result in results]).groupby(EVT_COL)
    print('### Performing triplets analysis on {0:d} events'.format(len(groups)))
    # Splitting event numbers into groups with different deviations from the trigger
    deviations = [0,2,4,6,8,10]
    event_deviations = {}
    for dev in deviations:
        event_deviations[dev] = []
    # Analysing each event
    n_events = len(groups)
    n_events_processed = 0
    for event, df in groups:
        n_events_processed += 1
        print_progress(n_events_processed, n_events)
        nHits = df.shape[0]
        # Selecting only hits in the acceptance region
        sel = pd.concat([(df['SL'] == sl) & (df['TDC_CHANNEL_NORM'].isin(ch)) 
                    for sl, ch in ACCEPTANCE_CHANNELS.items()], axis=1).any(axis=1)
        df = df[sel]
        nHitsAcc = df.shape[0]
        df_events.loc[event, ['HITS_MULT_ACCEPTED', 'HITS_MULT']] = (nHitsAcc, nHits)
        # Checking TIME0 found in each chamber
        tzeros = {}
        # print(event)
        # print(df[['SL', 'TDC_CHANNEL_NORM', 'LAYER', 'TIMENS']])
        time0 = df_events.loc[event, 'TIME0']
        for sl, df_sl in df.groupby('SL'):
            # print('--- SL: {0:d}'.format(sl))
            nLayers = len(df_sl.groupby('LAYER'))
            # Skipping chambers that don't have 3 layers of hits
            if nLayers < 3:
                continue
            tzeros_sl, angles_sl = meantimer_results(df_sl, verbose=False)
            if sl not in tzeros:
                tzeros[sl] = []
            tzeros[sl].extend(tzeros_sl)
            meantimers_info = results[sl][3]
            for name in ['t0_dev', 't0_angle', 'hit_angles_diff', 'hit_means_diff']:
                if name not in meantimers_info:
                    meantimers_info[name] = []
            meantimers_info['t0_dev'].extend([time0 - tzero for tzero in tzeros_sl])
            meantimers_info['t0_angle'].extend(angles_sl)
            # print(tzeros_sl)
            # if len(df_sl) < 4:
            #     continue
            # # q = df_sl.sort_values('LAYER')['TIME_ABS'].values[:4]
            # # tzero = (q[0] + q[3] + 3*q[1] + 3*q[2])/8.0 - 0.5*TDRIFT
            # # Skipping chambers that don't have 4 layers of hits
            # if nLayers  < 4:
            #     continue
            # # Calculating angles between pairs of hits in each chamber
            # hits = df_sl.set_index('LAYER')
            # br()
            # hits = hits['TIMENS'].reindex(range(1,4+1), fill_value=0).values
            # angles = np.array([hits[2] - hits[0], hits[1] - hits[3]]) / (ZCELL*2)
            # means = [0.5 * (TDRIFT - hits[0] + hits[3]), 0.5 * (TDRIFT - hits[2] + hits[1])]
            # meantimers_info['hit_angles_diff'].append(angles[1] - angles[0])
            # meantimers_info['hit_means_diff'].append(means[1] - means[0])
            # br()
        # Calculating the mean of the t0 candidates excluding outliers
        tzero, tzeros, nSLs = mean_tzero(tzeros)
        if len(tzeros) < 1:
            df_events.loc[event, 'MEANTIMER_MULT'] = 0
        else:
            df_events.loc[event, ['MEANTIMER_MEAN', 'MEANTIMER_MIN', 'MEANTIMER_MAX', 'MEANTIMER_MULT', 'MEANTIMER_SL_MULT']
            ] = (tzero, np.min(tzeros), np.max(tzeros), len(tzeros), nSLs)
        deviation = abs(tzero - time0)
        for dev in reversed(deviations):
            if deviation > float(dev):
                event_deviations[dev].append(event)
                break
        # print(tzero_range)
        # tzeros_a = np.sort(np.array(tzeros))
        # print(tzeros_a - time0)
        # print('TRG: {0:2f}   >><<   {1:2f} :MEANTIMER'.format(time0, tzero))
        # if not tzeros:
            # br()
        # print('### MEANTIMERS in event: {0:d}  with {1:d} layers'.format(event, df['SL'].nunique()))
        # print(sorted(np.array(tzeros) - np.mean(tzeros)))
    # # Listing all events grouped by deviation from the trigger time
    # for dev in deviations:
    #     print('+ DEVIATION: {0:d}'.format(dev))
    #     print(event_deviations[dev])
    # # Dropping events that have insufficient hits in all SLs
    # df_events.drop(df_events[df_events['HITS_MULT'] < 0].index, inplace=True)


def process(input_files):
    """Do the processing of input files and produce all outputs split into groups if needed"""
    jobs = []  
    results = []

    parts = os.path.split(input_files[0])
    run = os.path.split(parts[0])[-1]
    runnum = int(run.split('Run')[-1])

    if args.layer is None:
        # Processing all layers in parallel threads
        allhits, df_events = read_data(input_files, runnum)
        for sl in range(4):
        # Avoiding parallel processing due to memory duplication by child processes
            analyse(allhits, sl)
        # pool = Pool(4)
        # results = pool.map(analyse_parallel, jobs)
    else:
        # Running the analysis on SL 0
        allhits, df_events = read_data(input_files, runnum)
        analyse(allhits, args.layer)
    # Matching triplets from same event
    if args.triplets:
        # FIXME: This is not going to work due to the changed format of the function input
        sync_triplets(allhits, df_events)
    
    print('### Writing output')

    # Determining output file path
    file = os.path.splitext(parts[-1])[0]
    if args.events:
        file += '_e'+'_'.join(['{0:d}'.format(ev) for ev in args.events])
    if args.update_tzero:
        file += '_t0'
    if args.suffix:
        file += '_{0:s}'.format(args.suffix)

    ### GENERATE TEXT OUTPUT [one event per line]
    if args.root:
        out_path = os.path.join('text', run, file+'.txt')
        try:
            os.makedirs(os.path.dirname(out_path))
        except:
            pass
        save_root(allhits, df_events, out_path)

    ### GENERATE CSV OUTPUT
    if args.csv:
        out_path = os.path.join('text', run, file+'.csv')
        df_out = allhits[['EVENT_NR', 'FPGA', 'TDC_CHANNEL', 'SL','LAYER','TDC_CHANNEL_NORM', 'ORBIT_CNT', 'TIMENS', 'TIME0','X_POS_LEFT','X_POS_RIGHT','Z_POS']]
        print('### Writing {0:d} hits to file: {1:s}'.format(df_out.shape[0], out_path))
        df_out.to_csv(out_path)

    print('### Done')


for i in range(0, len(args.inputs), args.group):
    files = args.inputs[i:i+args.group]
    print('############### Starting processing files {0:d}-{1:d} out of total {2:d}'.format(i, i+len(files)-1, len(args.inputs)))
    # Processing the input files
    process(files)
