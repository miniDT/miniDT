#!/usr/bin/env python
from pdb import set_trace as br
from numpy.polynomial.polynomial import Polynomial
from modules.utils import OUT_CONFIG
from modules.geometry.hit import HitManager
from modules.geometry.sl import SL
from modules.geometry import Geometry
from modules.reco import config, plot
from modules.analysis import config as CONFIGURATION

import os
import itertools
import bokeh

############################################# INPUT ARGUMENTS 
import argparse
parser = argparse.ArgumentParser(description='Track reconstruction from input hits.')
parser.add_argument('-E', '--events', metavar='N',  help='Events to process', type=int, default=None, nargs='+')
parser.add_argument('-f', '--format',  help='Input hits format', default='time_wire')
parser.add_argument('-g', '--glance',  help='Only show # hits in each event', action='store_true', default=False)
parser.add_argument('-o', '--output',  help='Output path', action='store_true', default='plots/hits_ev{0:d}.html')
parser.add_argument('-p', '--plot',  help='Draw plots', action='store_true', default=False)
parser.add_argument('inputs', metavar='FILE', help='Input files with raw hits, 1 event/line', nargs='+')

args = parser.parse_args()
# Checking validity of the input format
if args.format not in OUT_CONFIG:
    raise ValueError('Wrong input format (-f) specified')
# Checking existence of input files
for file_path in args.inputs:
    if not os.path.exists(os.path.expandvars(file_path)):
        print('--- ERROR ---')
        print('  \''+file_path+'\' file not found')
        print('  please provide the correct path to the file containing raw hits' )
        print()
        exit()


def process(input_files):
    """Reconstruct tracks from hits in all events from the provided input files"""
    n_words_event = len(OUT_CONFIG['event']['fields'])
    n_words_hit = len(OUT_CONFIG[args.format]['fields'])
    # Initialising event
    event = -1
    G = Geometry(CONFIGURATION)
    H = HitManager()
    SLs = {}
    for iSL in config.SL_SHIFT.keys():
        SLs[iSL] = SL(iSL, config.SL_SHIFT[iSL], config.SL_ROTATION[iSL])
    # Defining which SLs should be plotted in which global view
    GLOBAL_VIEW_SLs = {
        'xz': [SLs[0], SLs[2]],
        'yz': [SLs[1], SLs[3]]
    }
    # Analyzing the hits in each event
    for file_path in input_files:
        # Reading input file line by line
        with open(file_path, 'r') as file_in:
            file_line_nr = 0
            for line in file_in:
                file_line_nr += 1
                if file_line_nr <= 1:
                    continue
                hits_lst = []
                H.reset()
                words = line.strip().split()
                event = int(words[0])
                # Skipping event if it was not specified in command line
                if args.events is not None and event not in args.events:
                    continue
                nhits = int(words[1])
                print('Event {0:<5d}  # hits: {1:d}'.format(event, nhits))
                if args.glance:
                    continue
                # Extracting hit information
                for iHit in range(nhits):
                    start = n_words_event + iHit*n_words_hit
                    ww = words[start:start+n_words_hit]
                    hits_lst.append([int(ww[0]), int(ww[1]), int(ww[2]), float(ww[3])])
                H.add_hits(hits_lst)
                # Calculating local+global hit positions
                H.calc_pos(SLs)
                # Creating figures of the chambers
                figs = {}
                figs['sl'] = plot.book_chambers_figure(G)
                figs['global'] = plot.book_global_figure(G, GLOBAL_VIEW_SLs)
                # Analyzing hits in each SL
                for iSL, sl in SLs.items():
                    hits_sl = H.hits.loc[H.hits['sl'] == iSL].sort_values('layer')
                    if args.plot:
                        # Drawing the left and right hits in local frame
                        figs['sl'][iSL].square(x=hits_sl['lposx'], y=hits_sl['posz'], size=5,
                                        fill_color='red', fill_alpha=0.7, line_width=0)
                        figs['sl'][iSL].square(x=hits_sl['rposx'], y=hits_sl['posz'], size=5,
                                        fill_color='blue', fill_alpha=0.7, line_width=0)
                    # Performing track reconstruction in the local frame
                    layer_groups = hits_sl.groupby('layer').groups
                    hitid_layers = [gr.to_numpy() for gr_name, gr in layer_groups.items()]
                    # Building the list of hit combinations from all available layers
                    hits_layered = list(itertools.product(*hitid_layers))
                if args.plot:
                    # Drawing the left and right hits in global frame
                    for view, sls in GLOBAL_VIEW_SLs.items():
                        sl_ids = [sl.id for sl in sls]
                        hits_sls = H.hits.loc[H.hits['sl'].isin(sl_ids)]
                        figs['global'][view].square(x=hits_sls['glpos'+view[0]], y=hits_sls['glpos'+view[1]],
                                                    fill_color='red', fill_alpha=0.7, line_width=0)
                        figs['global'][view].square(x=hits_sls['grpos'+view[0]], y=hits_sls['grpos'+view[1]],
                                                    fill_color='blue', fill_alpha=0.7, line_width=0)

                # Storing the figures to an HTML file
                if args.plot:
                    plots = [[figs['sl'][l]] for l in [3, 1, 2, 0]]
                    plots.append([figs['global'][v] for v in ['xz', 'yz']])
                    bokeh.io.output_file(args.output.format(event), mode='cdn')
                    bokeh.io.save(bokeh.layouts.layout(plots))



process(args.inputs)
