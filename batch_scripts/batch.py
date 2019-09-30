#!/bin/env python

import imp
import copy
import os
import subprocess
import glob
import argparse
from modules.utils import chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create batch jobs')
    parser.add_argument('jobpath', metavar='PATH', help='folder where jobs are created', nargs='?', default ="BATCH")
    parser.add_argument('-n', '--numFiles',  metavar='N', help='number of files per job', action='store', default=5)
    parser.add_argument('-p', '--path', help='Full path of lemma run list on eos', action='store', default='/eos/project/l/lemma/data/2019_LNL/')
    parser.add_argument('-r', '--run' , help='Run number to analyze', nargs='?', default ="Run000453")
    parser.add_argument('-q', '--queue' , help='Condor queue: espresso, microcentury, longlunch, workday, tomorrow, testmatch, nextweek', action='store', default="microcentury")

    args = parser.parse_args()

    # Getting input files
    inputpath = os.path.normpath(args.path+args.run)
    fullList = glob.glob(inputpath+"/data*.txt")
    # Splitting input files into jobs
    fileBlocks =  chunks(fullList, int(args.numFiles))
    # Constructing folder for job output
    outputpath = os.path.join(args.jobpath, args.run)
    i = 0
    for chunk, files in enumerate(fileBlocks) :
       jobDir = os.path.join(outputpath, "job_"+str(i))
       print('Creating output directory: {0}'.format(jobDir))
       path = jobDir
       os.makedirs(path)

       inputListFile= open(path+'/inputFiles.txt','w')
       inputListFile.write(" ".join(files))
       inputListFile.close()
       i += 1

    cmd = 'condor_submit condor_config.sub subDir=$PWD -append "+JobFlavour={0:s}" -batch-name {1:s} -queue jobDir matching dirs {2:s}/*'.format(args.queue, args.run, outputpath)
    subprocess.call(cmd, shell=True)
