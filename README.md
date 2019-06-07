### Analysis code to analyse data from miniDT chambers

To set up the working Python 3 environment, do:
   * `git clone https://github.com/miniDT/miniDT.git`
   * `cd miniDT/`
   * `virtualenv -p python3 ./` - create local isolated python environment, where all dependencies will be installed
   * `source setenv.sh` or `source bin/activate` - activate the local python environment
   * `pip install --upgrade pip` - update pip installer to the latest version
   * `pip install -r requirements.txt` - install all required packages

To profile the code:
   * `pip install gprof2dot` - *just for the first time*
   * `python -m cProfile -o profile.stats <command>`
   * `gprof2dot -f pstats profile.stats | dot -Tpng -o profile.png`

- - - - -

### Running the offline analysis
   * using raw external trigger:  
     `./offline_analysis.py -rep <list of input TXT files>`
   * using raw external trigger, with `t0` determined by meantimer (for avoiding jitter of the external trigger):  
     `./offline_analysis.py -urepa <list of input TXT files>`
   * using meantimer to find aligned hits in each orbit (external trigger not used at all):
     `./offline_analysis.py -trpa <list of input TXT files>`

Exclude the `-p` option if you don't need to produce the plots.  
Events with no hits are excluded.
