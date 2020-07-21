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

### Processing the RAW data to extract hits:

Configuration of the setup defined in `modules/analysis/config.py`.  
One of the following methods for event search can be used:
   * using raw external trigger:  
     `./process_hits.py -e <formats> <list of input TXT files>`
   * using raw external trigger, with `t0` determined by meantimer (for avoiding jitter of the external trigger):  
     `./process_hits.py -ea <formats> <list of input TXT files>`
   * using meantimer to find aligned hits in each orbit (external trigger not used at all):  
     `./process_hits.py -a <formats> <list of input TXT files>`
  
  Available format options are the ones starting with `--hits_*` in the help message  
  Use `-s <suffix>` to add different suffixes to the output text files depending on the used event search method

- - - - -

### Performing track reconstruction
Configuration of the track reconstruction defined in `modules/reco/config.py`.  
Output of the `process_hits.py` is used as input:  
`./reco_tracks.py <list of input TXT files>`
