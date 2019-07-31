"""Code for plotting during reconstruction"""

from bokeh.plotting import figure
from pdb import set_trace as br
from modules.geometry import DTYPE_COOR

import numpy as np

PLOT_RANGE = {'x': (0, 800), 'y': (0, 1000)}

def book_chambers_figure(geo):
    """Book figures for plotting the hits inside chambers"""
    figs = {}
    for sl in geo.SLS:
        cell_borders = geo.cell_borders()
        wire_positions = geo.wire_positions()
        # Creating the figure
        fig = figure(
            plot_width=2520, plot_height=195,
            x_range=[geo.SL_FRAME['l'], geo.SL_FRAME['r']],
            y_range=[geo.SL_FRAME['b'], geo.SL_FRAME['t']],
            title="Local SL {0:d}".format(sl),
            x_axis_label="x (mm)",
            y_axis_label="y (mm)")
        # Drawing the cells layout
        for l in range(1, geo.NLAYERS+1):
            fig.quad(left=geo.CELL_BORDERS[l]['l'], right=geo.CELL_BORDERS[l]['r'], 
                     bottom=geo.CELL_BORDERS[l]['b'], top=geo.CELL_BORDERS[l]['t'],
                     line_color='slategray', fill_color='gray', fill_alpha=0.1)
        # Drawing wires
        for l in range(1, geo.NLAYERS+1):
            fig.asterisk(x=geo.WIRE_POSITIONS[l]['x'], y=geo.WIRE_POSITIONS[l]['z'],
                         line_color='gray', line_alpha=0.4, size=6)
        figs[sl] = fig
    return figs


def book_global_figure(geo, sls):
    """Book figures for plotting global hit positions in two orthogonal planes"""
    figs = {}
    figs['xz'] = figure(plot_width=1250, plot_height=1300,
                        x_range=PLOT_RANGE['x'], y_range=PLOT_RANGE['y'],
                        title="Global XZ",
                        x_axis_label='x (mm)',
                        y_axis_label='z (mm)')
    figs['yz'] = figure(plot_width=1250, plot_height=1300,
                        x_range=PLOT_RANGE['x'], y_range=PLOT_RANGE['y'],
                        title="Global YZ",
                        x_axis_label='y (mm)',
                        y_axis_label='z (mm)')
    # Drawing SL frames
    for name in ['xz', 'yz']:
        for sl in sls[name]:
            # Building the array of XYZ coordinates of the four chamber's corners
            frame_local = np.array([
               [geo.SL_FRAME['l'], 0, geo.SL_FRAME['t']],
               [geo.SL_FRAME['r'], 0, geo.SL_FRAME['t']],
               [geo.SL_FRAME['r'], 0, geo.SL_FRAME['b']],
               [geo.SL_FRAME['l'], 0, geo.SL_FRAME['b']],
            ], dtype=DTYPE_COOR)
            frame_global = sl.coor_to_global(frame_local)
            fig = figs[name]
            # Extracting X/Y coordinates for plotting depending on the view plane
            frame_y = frame_global[:, 2]  # Z -> Y
            frame_x = None
            if name == 'xz':
                frame_x = frame_global[:, 0]  # X -> X
            elif name == 'yz':
                frame_x = frame_global[:, 1]  # Y -> X
            # Drawing a line representing the frame of the superlayer
            fig.patch(x=frame_x, y=frame_y, 
                     line_color='slategray', fill_color='gray', fill_alpha=0.2)
            # Drawing a text with the SL id
            fig.text(x=[50], y=[frame_y.max()+20], text=['SL{0:d}'.format(sl.id)],
                     text_align='left', text_color='slategray')
    return figs

