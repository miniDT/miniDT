"""Preparing the plots"""

import os
import numpy as np
import pandas as pd
from bokeh.io import save, output_file
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import formatters
from pdb import set_trace as br

from .config import NCHANNELS, XCELL, ZCELL, VDRIFT, DURATION

SL_CHANNELS = [(49, 64+1), (49, 64+1), (1, 16+1), (1, 16+1)]

class DataPlots():
    """Collection of cumulative plots with data storage for potential merging of multiple runs"""

    PLOT_SIZE = (450, 300)

    def __init__(self):
        self.book_plots()

    def figs(self, names=[]):
        """Returns a list of figures according to the input names"""
        return [self.FIGURES[name] for name in names]
    
    def add(self, name, title='', x_title='', y_title=''):
        """Creates a Bokeh figure for the plot"""
        self.FIGURES[name] = figure(
            plot_width = self.PLOT_SIZE[0],
            plot_height = self.PLOT_SIZE[1],
            title = title,
            y_axis_label = y_title,
            x_axis_label = x_title
        )

    def fill_hist(self, name, hist, edges, add=False):
        """Fills using binned histogram format"""
        # Creating histogram from the data
        if name in self.FIGURES:
            self.FIGURES[name].quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
        # Storing data
        sel = self.DATA['name'].str.match(name+'$')
        if not sel.any():
            xl = edges[:-1]
            xh = edges[1:]
            y = hist
            bins = range(len(hist))
            names = [name] * len(hist)
            self.DATA = self.DATA.append(pd.DataFrame({
                'name': names, 'bin': bins, 'xl': xl, 'xh': xh, 'y': y
            }), ignore_index=True)
        else:
            if add:
                self.DATA.loc[sel, 'y'] = self.DATA[sel]['y'] + hist
            else:
                self.DATA.loc[sel, 'y'] = hist

    def fill(self, name, data, bins, bins_range=None):
        """Stores the histogram in a figure and data storage"""
        hist, edges = np.histogram(data, density=False, bins=bins, range=bins_range)
        self.fill_hist(name, hist, edges, add=False)

    def book_plots(self):
        """Creates all figures to be filled"""
        self.FIGURES = {}
        self.DATA = pd.DataFrame({
            'name': ['test'],
            'bin': np.array([0], dtype=np.int8),
            'xl': np.array([0.0], dtype=np.float32),
            'xh': np.array([1.0], dtype=np.float32),
            'y':  np.array([1], dtype=np.int8),
        })
        for sl in range(4):
            self.add('tdcchan_SL{0:d}'.format(sl), 'channel: SL{0:d}'.format(sl), 'channel', '# hits')
            self.add('tdcmeas_SL{0:d}'.format(sl), 'TDC count: SL{0:d}'.format(sl), 'TDC count', 'hits')
            self.add('t0_diff_SL{0:d}'.format(sl), 't0 difference: SL{0:d}'.format(sl), 't0 difference for multiple triplets', 'triplets')
            self.add('t0_dev_SL{0:d}'.format(sl), 'trigger - t0: SL{0:d}'.format(sl), 'time difference [ns]', 'triplets')
            self.add('t0_angle_SL{0:d}'.format(sl), 't0 angle: SL{0:d}'.format(sl), 'triplet angle [rad]', 'triplets')
            self.add('t0_mult_SL{0:d}'.format(sl), 't0 multiplicity: SL{0:d}'.format(sl), 'multiplicity of t0 evaluated', 'events')
            self.add('hit_means_diff_SL{0:d}'.format(sl), 'hit-pair means difference: SL{0:d}'.format(sl), 'time difference', 'events')
            self.add('hit_angles_diff_SL{0:d}'.format(sl), 'hit-pair angles difference: SL{0:d}'.format(sl), 'angles difference', 'events')
            self.add('triplet_angle_SL{0:d}'.format(sl), 'triplet angle: SL{0:d}'.format(sl), 'angle [rad]', 'triplets')
            self.add('timebox_SL{0:d}'.format(sl), 'time: SL{0:d}'.format(sl), 'time [ns]', 'hits')
        self.add('cell_nhits_max', 'Max # hits in a single cell', '# hits', 'events')
        self.add('cell_dt_min', 'Min dt between 2 hits in a single cell', 'time [ns]', 'events')
        self.add('cell_dt_max', 'Max dt between 2 hits in a single cell', 'time [ns]', 'events')
        # # Adding timebox for each channel of each chamber
        # for sl, chs in enumerate(SL_CHANNELS):
        #     for ch in range(*chs):
        #         self.add('timebox_SL{0:d}_CH{1:d}'.format(sl, ch), 'time - SL{0:d} | CH{1:d}'.format(sl, ch), 'time [ns]', 'hits')
        # Event-wise distributions
        self.add('trg_bits', 'presence of trigger signal', 'trigger bits: 2-(0,130) 1-(0,129) 0-(1,129)', 'events')
        self.add('trg_meantimer', 'trigger - mean(meantimers)', 'time [ns]', 'events')
        self.add('meantimer_mult', 'meantimer cluster size', '# triplets', 'events')
        self.add('meantimer_sl_mult', 'number of SLs in the meantimer cluster', '# Sls', 'events')
        self.add('meantimer_dev_max', 'max meantimer deviation', 'max(|t0 - mean(t0)|) [ns]', 'events')
        self.add('hits_mult', 'hits multiplicity', '# hits', 'events')
        self.add('time_event', 'min time from previous/next event', '<1000: time [ns]; >1000: orbits [{0} ns]'.format(DURATION['orbit']), 'events')
        for bits in [(2,0), (2,1)]:
            self.add('trg_time_{0:d}{1:d}'.format(bits[0], bits[1]), 
                     'trigger time delay: bit {0:d} -> {1:d}'.format(bits[0], bits[1]), 
                     'time [ns]', 'events')

    def save(self, output_path):
        """Saves the figure content to a CSV file"""
        self.DATA.drop(self.DATA[self.DATA['name'] == 'test'].index, inplace=True)
        self.DATA.to_csv(output_path, float_format='%.2f', index=False)

    def read(self, input_path):
        """Reads figures content from a CSV file"""
        data = pd.read_csv(input_path, engine='c')
        for name, df in data.groupby('name'):
            edges = df['xl']
            edges = edges.append(df['xh'].iloc[-1:])
            hist = df['y']
            # br()
            self.fill_hist(name, hist.tolist(), edges.tolist(), add=True)

    def generate_figures(self):
        """Generate new figures form accumulated histogram data"""
        for name, fig in self.FIGURES.iteritems():
            sel = self.DATA['name'].str.match(name)
            if not sel.any():
                continue
            hist = self.DATA[sel]['y']
            left = self.DATA[sel]['xl']
            right = self.DATA[sel]['xh']
            self.FIGURES[name].quad(top=hist, bottom=0, left=left, right=right)

### DEFINE THE STRUCTURE HOLDING FIGURES AND PLOT DATA
DP = DataPlots()

### DEFINE BOKEH FIGURE (CANVAS EQUIVALENT)
# cell grid - used for plotting
grid_b = []
grid_t = []
grid_l = []
grid_r = []
for lay in [1,2,3,4]:
    for cell in range(1,int(NCHANNELS/4)+1):
        grid_b.append( 4*ZCELL - lay * ZCELL )
        grid_t.append( grid_b[-1] + ZCELL )
        grid_l.append( (cell-1) * XCELL)
        grid_r.append( grid_l[-1] + XCELL )
        if lay%2 == 0:
            grid_l[-1]  += XCELL/2.
            grid_r[-1] += XCELL/2.

# channel-layer map
def map(chan):
  mod = chan%4
  if mod == 1:
    return chan, 1
  elif mod == 2:
    return chan-1., 3
  elif mod == 3:
    return chan-0., 2
  else: # mod == 0:
    return chan-1., 4

p_chanrate_SL = {} # hit rate of each channel
p_hitsperorbitnumber_SL = {} # nHits per orbit vs orbit number
p_orbdiffperorbitnumber_SL = {} # hits orbit difference vs orbit number
p_occ_SL = {} # occupancy
p_pos_SL = {} # hit position in chamber

for SL in range(0,4):
    p_chanrate_SL[SL] = figure(
        plot_width=450,
        plot_height=300,
        title="channel - SL%d" % SL,
        y_axis_label="rate [Hz]",
        x_axis_label="channel",
    )


    p_hitsperorbitnumber_SL[SL] = figure(
        plot_width=900,
        plot_height=300,
        title="hits per orbit - SL%d" % SL,
        y_axis_label="hits",
        x_axis_label="orbit #",
        output_backend="webgl",
    )
    p_hitsperorbitnumber_SL[SL].xaxis[0].formatter = formatters.PrintfTickFormatter(format="%d")

    p_orbdiffperorbitnumber_SL[SL] = figure(
        plot_width=900,
        plot_height=300,
        title="orbit difference per orbit - SL%d" % SL,
        y_axis_label="difference wrt previous event",
        x_axis_label="time [ns]",
        output_backend="webgl",
    )
    p_orbdiffperorbitnumber_SL[SL].xaxis[0].formatter = formatters.PrintfTickFormatter(format="%d")

    p_occ_SL[SL] = figure(
        plot_width=450,
        plot_height=300,
        y_range=[4.5,0.5],
        x_range=[-1,NCHANNELS+1],
        title="occupancy - SL%d" % SL,
        y_axis_label="layer",
        x_axis_label="channel",
    )

    # x_ranges = [(500, 700), (500, 700), (-10, 190), (-10, 190)]
    x_ranges = [(400, 700), (400, 700), (-10, 290), (-10, 290)]
    p_pos_SL[SL] = figure(
        plot_width=1800,
        plot_height=120,
        y_range=[0,60],
        x_range=[-XCELL/2,XCELL*(NCHANNELS/4+1)],
        # x_range=[40, 86],
        # x_range=x_ranges[SL],
        title="hit position - SL%d" % SL,
        y_axis_label="y (mm)",
        x_axis_label="x (mm)",
    )


def fill_plots(SL, dfhits, df=None, meantimer_info=None, df_events=None, event_col='ORBIT_CNT'):
    """Prepares plots using input dataframes"""
    # Selecting only physical hits for plots
    dfhits = dfhits.loc[dfhits['TDC_CHANNEL_NORM'] <= NCHANNELS]

    # Filling plots for all input hits
    if not dfhits.empty:
        DP.fill('tdcchan_SL{0:d}'.format(SL), dfhits['TDC_CHANNEL_NORM'], range(0, NCHANNELS+2))
        DP.fill('tdcmeas_SL{0:d}'.format(SL), dfhits['TDC_MEAS'], range(0, 32))
        # Converting entries to rates
        hist, edges = np.histogram(dfhits['TDC_CHANNEL_NORM'], density=False, bins=range(0, NCHANNELS+2))
        duration = dfhits['TIME_ABS'].max() - dfhits['TIME_ABS'].min()
        hist = hist * 1e9 / max(duration, 1.0)
        p_chanrate_SL[SL].quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])

        # Occupancy plots
        occchan = [map(c)[0] for c in range(1, NCHANNELS+1)]
        occlay  = [map(c)[1] for c in range(1, NCHANNELS+1)]
        occ     = []
        somecolors = []
        maxcount = float(max(hist))
        for c in range(1, NCHANNELS+1):
            cval = dfhits['TDC_CHANNEL_NORM'][dfhits['TDC_CHANNEL_NORM']==c].count()/maxcount if maxcount>0 else dfhits['TDC_CHANNEL_NORM'][dfhits['TDC_CHANNEL_NORM']==c].count()
            occ.append(cval)
            somecolors.append("#%02x%02x%02x" % (int(255*(1-cval)), int(255*(1-cval)), int(255*(1-cval))) if cval>0 else '#ffffff')
        # Orbit differences
        orbit_counts = dfhits.groupby('ORBIT_CNT', as_index=False).agg('size')
        p_hitsperorbitnumber_SL[SL].square(
            x = orbit_counts.index.tolist(),
            y = orbit_counts.tolist(),
            size=5,
        )
        p_orbdiffperorbitnumber_SL[SL].square(
            x = dfhits['ORBIT_CNT'].tolist(),
            y = dfhits['ORBIT_CNT'].diff().tolist(),
            size=5,
        )
        p_occ_SL[SL].scatter(x=occchan, y=occlay, fill_color=somecolors, 
                             marker='square', size=12, line_color='black')

    # Filling plots for events with reconstructed triplets or external trigger
    if df is not None and not df.empty:
        DP.fill('timebox_SL{0}'.format(SL), df['TIMENS'], 1000, (-300,700))
        # # Filling timebox of each channel of this layer
        # for ch in range(*SL_CHANNELS[SL]):
        #     DP.fill('timebox_SL{0:d}_CH{1:d}'.format(SL, ch), df.loc[df['TDC_CHANNEL_NORM'] == ch, 'TIMENS'], 150, (-50,100))

        p_pos_SL[SL].quad(top=grid_t,  bottom=grid_b,  left=grid_l, right=grid_r,  
                          fill_color='white', line_color='black')
        p_pos_SL[SL].scatter(x=df['X_POS_RIGHT'], y=df['Z_POS'], marker='square', size=2)
        p_pos_SL[SL].scatter(x=df['X_POS_LEFT'], y=df['Z_POS'], marker='square', size=2)

        colors = ["red", "palevioletred", "darkorange", "gold", "dodgerblue", "slateblue",  "mediumseagreen", "lightgreen"]
        events = df_events.index.values
        for iO, event in enumerate(events[-8:]):
            q = df[(df['EVENT_NR'] == event)]
            color = colors[iO]
            p_occ_SL[SL].scatter(x=q['TDC_CHANNEL_NORM']+q['X_CHSHIFT'], y=q['LAYER'], 
                                 marker='square', size=12, line_color=color, fill_color=color)
            p_pos_SL[SL].scatter(x=q['X_POS_LEFT'], y=q['Z_POS'], marker='square', 
                                 line_color=color, fill_color=color, size=5)
            p_pos_SL[SL].scatter(x=q['X_POS_RIGHT'], y=q['Z_POS'], marker='square', 
                                 line_color=color, fill_color=color, size=5)

    # Filling plots for book-keeping
    if meantimer_info is not None:
        DP.fill('t0_diff_SL{0:d}'.format(SL), meantimer_info['t0_diff'], 200, (-100,100))
        DP.fill('t0_mult_SL{0:d}'.format(SL), meantimer_info['t0_mult'], 30, (0,30))
        DP.fill('triplet_angle_SL{0:d}'.format(SL), meantimer_info['triplet_angle'], 300, (3,3))
        if 't0_dev' in meantimer_info:
            # DP.fill('t0_dev_SL{0:d}'.format(SL), meantimer_info['t0_dev'], 200, (-100,100))
            DP.fill('t0_dev_SL{0:d}'.format(SL), meantimer_info['t0_dev'], 60, (-30,30))
        if 't0_angle' in meantimer_info:
            # DP.fill('t0_angle_SL{0:d}'.format(SL), meantimer_info['t0_angle'], 200, (-1,1))
            DP.fill('t0_angle_SL{0:d}'.format(SL), meantimer_info['t0_angle'], 120, (-0.6,0.6))
        if 'hit_angles_diff' in meantimer_info:
            DP.fill('hit_angles_diff_SL{0:d}'.format(SL), meantimer_info['hit_angles_diff'], 400, (-1.0,1.0))
        if 'hit_means_diff' in meantimer_info:
            DP.fill('hit_means_diff_SL{0:d}'.format(SL), meantimer_info['hit_means_diff'], 400, (-20,20))

    # Filling plots with 1 entry/event
    if df_events is not None and not df_events.empty:
        DP.fill('trg_time_20', df_events['TIMEDIFF_TRG_20'], 250, (700,750))
        DP.fill('trg_time_21', df_events['TIMEDIFF_TRG_21'], 250, (100,150))
        DP.fill('trg_bits', df_events['TRG_BITS'], 9, (-1,8))
        evt_time = np.min(np.abs(df_events[['TIME0_BEFORE', 'TIME0_AFTER']]), axis=1)
        evt_time.loc[evt_time > 1000] = evt_time[evt_time > 1000] / DURATION['orbit'] + 1000
        DP.fill('time_event', evt_time, 5000, (0,2500))
        # DP.fill('cell_nhits_max', df_events['CELL_HITS_MULT_MAX'], 10, (0,10))
        # DP.fill('cell_dt_min', df_events['CELL_HITS_DT_MIN'], 1200, (0,600))
        # DP.fill('cell_dt_max', df_events['CELL_HITS_DT_MAX'], 1200, (0,600))
        if 'MEANTIMER_MULT' in df_events.columns:
            DP.fill('hits_mult', df_events['HITS_MULT_ACCEPTED'], 201, (-1,200))
            DP.fill('meantimer_mult', df_events['MEANTIMER_MULT'], 61, (-1,60))
            DP.fill('meantimer_sl_mult', df_events['MEANTIMER_SL_MULT'], 6, (-1,5))
            # Time difference between trigger signal and meantimer results
            values = df_events['TIME0'] - df_events['MEANTIMER_MEAN']
            values = values[df_events['MEANTIMER_MULT'] > 0]
            # DP.fill('trg_meantimer', values, 200, (-50,50))
            DP.fill('trg_meantimer', values, 60, (-15,15))
            # Meantimer deviation from other meantimers
            values_max = df_events['MEANTIMER_MAX'] - df_events['MEANTIMER_MEAN']
            values_min = df_events['MEANTIMER_MEAN'] - df_events['MEANTIMER_MIN']
            values = pd.concat([values_max, values_min], axis=1)
            DP.fill('meantimer_dev_max', values.max(axis=1), 250, (0, 50))


def draw_plots(output_path, occupancy=False, triplets=False, triggers=False, double_hits=False):
    """Draws plots into an output file"""
    print('### Writing plots to file: {0:s}'.format(output_path))
    output_file(output_path, mode="inline")

    ## SHOW OUTPUT IN BROWSER
    if occupancy:
        save(gridplot([[p_chanrate_SL[sl] for sl in range(4)],                      # channel multiplicity per SL
                      DP.figs(['tdcmeas_SL{0:d}'.format(sl) for sl in range(4)]),   # TDC meas multiplicity per SL
                      [p_occ_SL[sl] for sl in range(4)],                            # occupancy per SL
                      [p_hitsperorbitnumber_SL[0], p_hitsperorbitnumber_SL[1],],
                      [p_hitsperorbitnumber_SL[2], p_hitsperorbitnumber_SL[3],],    # multiplicity of hits per orbit
                    ]))
        return
    plots = [
        DP.figs(['tdcchan_SL{0:d}'.format(sl) for sl in range(4)]),                   # channel multiplicity per SL
        DP.figs(['tdcmeas_SL{0:d}'.format(sl) for sl in range(4)]),                   # TDC meas multiplicity per SL
        # [p_occ_SL[0], p_occ_SL[1], p_occ_SL[2], p_occ_SL[3]],                         # occupancy per SL
        DP.figs(['timebox_SL{0:d}'.format(sl) for sl in range(4)]),                   # timebox per SL
    ]
    # for sl, chs in enumerate(SL_CHANNELS):
    #      plots.append(DP.figs(['timebox_SL{0:d}_CH{1:d}'.format(sl, ch) for ch in range(*chs)]))    # timebox per channel

    if double_hits:
        plots.append(DP.figs(['cell_nhits_max', 'cell_dt_min', 'cell_dt_max']))   # info about multiple hits in a single cell
    if triggers:
        # plots.append(DP.figs(['time_event', 'trg_bits', 'trg_time_20', 'trg_time_21']))   # time delay between trigger signals
        if triplets:
            plots.append(DP.figs(['hits_mult', 'meantimer_mult', 'meantimer_sl_mult', 'trg_meantimer']))   # Meantimer information wrt trigger
            plots.append(DP.figs(['t0_dev_SL{0:d}'.format(sl) for sl in range(4)]))           # time delay between trigger and triplet t0
            plots.append(DP.figs(['t0_angle_SL{0:d}'.format(sl) for sl in range(4)]))         # angle of each candidate triplet
            # plots.append(DP.figs(['hit_angles_diff_SL{0:d}'.format(sl) for sl in range(4)]))         # angle difference between AC and BD hit pairs
            # plots.append(DP.figs(['hit_means_diff_SL{0:d}'.format(sl) for sl in range(4)]))         # center difference between AD and BC hit pairs

    if not triggers:
        if triplets:
            plots.extend([
                DP.figs(['t0_diff_SL{0:d}'.format(sl) for sl in range(4)]),             # difference of meantimer solution (t0) in case of multiple solutions
                DP.figs(['t0_mult_SL{0:d}'.format(sl) for sl in range(4)]),             # multiplicity of meantimer solutions
                DP.figs(['triplet_angle_SL{0:d}'.format(sl) for sl in range(4)]),       # angle of each potential meantimer solution
            ])
    # plots.extend([
    #     [p_hitsperorbitnumber_SL[0], p_hitsperorbitnumber_SL[1],],
    #     [p_hitsperorbitnumber_SL[2], p_hitsperorbitnumber_SL[3],],                          # multiplicity of hits per orbit
    # ])

    plots.extend([
        # [p_orbdiffperorbitnumber_SL[0], p_orbdiffperorbitnumber_SL[1],],
        # [p_orbdiffperorbitnumber_SL[2], p_orbdiffperorbitnumber_SL[3],],                    # orbit difference between hits
        [p_pos_SL[0],],
        [p_pos_SL[1],],
        [p_pos_SL[2],],
        [p_pos_SL[3],],                                                                     # hits position
    ])

    save(gridplot(plots))

    # Storing figure contents in CSV for potential merging later
    output_path = os.path.splitext(output_path)[0] + '.csv'
    print('### Writing figures content to file: {0:s}'.format(output_path))
    DP.save(output_path)


