#!/usr/bin/env python
"""Evaluate statistical distributions of the hits and events from CSV output of process_hits.py"""

import pandas as pd
import numpy as np
import os

from pdb import set_trace as br
from modules.analysis.config import NCHANNELS
from modules.utils import print_progress

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import save, output_file

SX = 350
SY = 250
FIGURES = {}

def read_csv(input_files):
    """Reads CSV data into a dataframe of the proper format"""
    dfs = []
    for file_in in input_files:
        dfs.append(pd.read_csv(file_in))
    return pd.concat(dfs)


def book_figures():
    """Create the Bokeh figures"""
    FIGURES['timebox'] = {}
    FIGURES['occupancy'] = {}
    for sl in range(4):
        FIGURES['timebox'][sl] = figure(
            plot_width=SX,
            plot_height=SY,
            title='Timebox [SL {0:d}]'.format(sl),
            y_axis_label="Hits",
            x_axis_label="Time (ns)",
        )
        FIGURES['occupancy'][sl] = figure(
            plot_width=SX,
            plot_height=SY,
            title="Occupancy [SL {0:d}]".format(sl),
            y_axis_label="Hits",
            x_axis_label="Channel",
        )
    FIGURES['n_hits'] = figure(
        plot_width=SX,
        plot_height=SY,
        title="Hits multiplicity".format(sl),
        y_axis_label="Events",
        x_axis_label="# hits",
    )


def fill_figures(hits_all):
    """Perform the analysis of all hits"""
    # Grouping hits by the event number
    hits = hits_all[hits_all['TDC_CHANNEL_NORM'] <= NCHANNELS]
    evt_groups = hits.groupby(hits_all['EVENT_NR'])
    # Filling plots per SL
    for sl in range(4):
        hits_sl = hits.loc[hits['SL'] == sl]
        # Filling the timebox
        v, e = np.histogram(hits_sl['TIMENS'], density=False, bins=300, range=(-200,400))
        FIGURES['timebox'][sl].quad(top=v, bottom=0, left=e[:-1], right=e[1:])
        v, e = np.histogram(hits_sl['TDC_CHANNEL_NORM'], density=False, bins=65, range=(0,65))
        FIGURES['occupancy'][sl].quad(top=v, bottom=0, left=e[:-1], right=e[1:])
    # Filling aggregated plots for all events
    v, e = np.histogram(evt_groups.agg('size'), density=False, bins=100, range=(0,200))
    FIGURES['n_hits'].quad(top=v, bottom=0, left=e[:-1], right=e[1:])
    # Creating a dataframe with 1 row per event
    df_events = pd.DataFrame(data={'EVENT_ID': list(evt_groups.groups.keys())})
    df_events['TIME0'] = -1
    df_events['TIMEDIFF_EXT0'] = -1e9
    df_events['TIMEDIFF_EXT1'] = -1e9
    df_events['TIMEDIFF_INT10'] = -1e9
    # Calculating time of the different trigger signals
    for event, df in evt_groups:
        df = df.sort_index()

    # n_events = len(evt_groups)
    # n_events_done = 0
    # for event, hits in evt_groups:
    #     print_progress(n_events_done, n_events)
    #     n_events_done += 1
    #     hits = hits.sort_index()
    #     br()

def draw_figures(output_path):
    """Arrange and draw the figures to an HTML file"""
    print('### Saving plots to: {0:s}'.format(output_path))
    plots = []
    for plot in ['timebox', 'occupancy']:
        plots.append([FIGURES[plot][sl] for sl in range(4)])
    plots.append([FIGURES['n_hits']])
    output_file(output_path, mode="inline")
    save(gridplot(plots))


############################################# INPUT ARGUMENTS 
import argparse
parser = argparse.ArgumentParser(description='Statistical evaluation of the hits and events extracted from the raw data.')
parser.add_argument('inputs', metavar='FILE', help='List of input CSV files', nargs='+')
args = parser.parse_args()

book_figures()
hits_all = read_csv(args.inputs)
fill_figures(hits_all)
output_path = os.path.splitext(args.inputs[0])[0] + '.html'
draw_figures(output_path)
