#!/bin/env python

"""Check contents of folders with jobs output to detect jobs that have succeded or failed"""

import os
import glob
import fnmatch
import subprocess

from modules.utils import print_progress, chunks
from pdb import set_trace as br

STATUSES = {
    -1: 'SETUP',
    0:  'SUBMITTED',
    1:  'DONE',
    2:  'KILLED',
    3:  'FAILED python',
    4:  'FAILED batch',
    5:  'DONE EMPTY',
    6:  'FAILED batch wall time',
}
FNULL = open(os.devnull, 'w')

def check_job(folder):
    """Checks a single job folder"""
    if not os.path.isfile(os.path.join(folder, 'out.log')):
      return -1
    file_log = os.path.join(folder, 'log.txt')
    # Check whether log file was created
    if os.path.isfile(file_log):
        # Check whether job is done
        if subprocess.call(['grep', '-E', '^### Done', file_log], stdout=FNULL, stderr=subprocess.STDOUT) == 0:
            return 1
    # Checking for file with the exit code
    file_exit = os.path.join(folder, 'exitStatus.txt')
    if os.path.isfile(file_exit):
        exit_code = int(subprocess.check_output(['head', file_exit]))
        return (4, exit_code)
    file_core = glob.glob(os.path.join(folder, 'core.*'))
    if len(file_core) > 0:
        return 4
    log_job = os.path.join(folder, 'out.log')
    if subprocess.call(['grep', 'Job terminated', log_job], stdout=FNULL, stderr=subprocess.STDOUT) == 0:
    # Checking the job output file
        file_job = os.path.join(folder, 'out.txt')
        if os.path.isfile(file_job):
            if subprocess.call(['grep', '-E', '^\+ Done', file_job], stdout=FNULL, stderr=subprocess.STDOUT) == 0:
                return 5
            # Getting the exit code from the job log file
            try:
                result = subprocess.check_output(['grep', '-E', '^Finished running:', file_job])
                exit_code = int(result.split()[-1])
                return (3, exit_code)
            except:
                return 4 
        elif subprocess.call(['grep', 'wall time', log_job], stdout=FNULL, stderr=subprocess.STDOUT) == 0:
            return 6

    return 0    

def check_jobs(input_folder='BATCH'):
    """Checks all jobs found in the input folder or deeper"""
    folders = []
    statuses = {}
    # Finding all input folders
    print('### Walking through subfolders of: {0:s}'.format(input_folder))
    for root, dirnames, filenames in os.walk(input_folder, topdown=True):
    # Skipping folders with split job output
        if os.path.split(root)[-1] in ['DONE', 'FAILED', 'OUTPUT']:
            dirnames[:] = []
            continue
        # Checking each job
        for dirname in fnmatch.filter(dirnames, 'job_*'):
            folder = os.path.normpath(os.path.join(root, dirname))
            folders.append(folder)
    folders.sort()
    # Checking each job
    print('### Checking {0:d} jobs:'.format(len(folders)))
    n_folders = len(folders)
    n_checked = 0
    for folder in folders:
        print_progress(n_checked, n_folders)
        status = check_job(folder)
        statuses[folder] = status
        n_checked += 1
    print('- - - - - - - - - - - - -')
    return statuses

def status_str(status):
    """Converts a status object to a printable string"""
    return STATUSES[status] if type(status) is int else '{0:s} << exit code: {1:d}'.format(STATUSES[status[0]], status[1])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check status of the submitted batch jobs')
    parser.add_argument('path', metavar='PATH', help='folder where jobs were created', nargs='?', default="BATCH")
    parser.add_argument('-c', '--clean', help='clean folders of failed jobs for resubmission', action='store_true', default=False)
    parser.add_argument('-q', '--queue', metavar='QUEUE', help='Resubmission queue: espresso, microcentury, longlunch, workday, tomorrow, testmatch, nextweek', 
            action='store', default='microcentury')
    parser.add_argument('-r', '--resubmit', help='resubmit runs with all jobs in SETUP state', action='store_true', default=False)
    parser.add_argument('-s', '--separate', help='separate jobs that are done/failed into different subfolders', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', help='show status of each job', action='store_true', default=False)
    args = parser.parse_args()

    # Running the checker on the input folder
    statuses = check_jobs(args.path)
    # Grouping folders by runs
    root_dir = None
    folders = {}
    for path in statuses.keys():
        parent = os.path.dirname(path)
        folder = os.path.basename(path)
        if parent not in folders:
            folders[parent] = []
        folders[parent].append(folder)
    # Sorting folders for each run
    for run in folders.keys():
        root_dir = os.path.dirname(run)
        folders[run].sort()
    for run in sorted(folders.keys()):
        run_statuses = {}
        for job in folders[run]:
            path = os.path.join(run, job)
            status = statuses[path]
            if status not in run_statuses:
                run_statuses[status] = []
            run_statuses[status].append(job)
        print(run)
        run_name = os.path.basename(run)
        # Resubmitting jobs
        if args.resubmit:
            # Resubmitting only runs that have all jobs in SETUP state
            if run_statuses.keys() != [-1]:
                continue
            # Resubmitting all jobs of the run
            runName = os.path.split(run)[-1]
            cmd = 'condor_submit condor_config.sub subDir=$PWD -append "+JobFlavour={0:s}" -batch-name {1:s} -queue jobDir matching dirs {2:s}/*'.format(args.queue, runName, run)
            if subprocess.call(cmd, shell=True) == 0:
                print('Resubmitted run: {0:s}'.format(run))
            else:
                print('WARNING: Failed to resubmit run: {0:s}'.format(run))
                print('- - - - - - - - - - - - - -')
                continue
        for status, jobs in run_statuses.items():
            print('   {0:s}  x  {1:d} jobs'.format(status_str(status), len(jobs)))
            if args.verbose:
                print('      '+' '.join(jobs))
            # Separating jobs into different subfolders if requested
            status_int = status if type(status) is int else status[0]
            if args.separate and status_int in (1, 2, 3, 4, 6):
                # Copying output files to a subfolder with just output files
                if status == 1:
                    out_dir_run = os.path.join(root_dir, 'OUTPUT', run_name+'/')
                    try:
                        os.makedirs(out_dir_run)
                    except OSError:
                        pass
                    command = ['find', run, '-name', 'data_*.txt', '-exec', 'cp', '{}', out_dir_run, ';']
                    # Executing the copy command
                    subprocess.call(command)
                # Splitting jobs to different subfolders
                subdir = None
                if status == 1:
                    subdir = os.path.join(root_dir, 'DONE')
                else:
                    subdir = os.path.join(root_dir, 'FAILED')
                # Creating the proper output folder for the run
                out_dir_run = os.path.join(subdir, run_name) + '/'
                try:
                    os.makedirs(out_dir_run)
                except OSError:
                    pass
                # Construction the command for moving all job folders of the run
                for jobs_chunk in chunks(jobs, 100):
                    n_moved = 0
                    command = ['mv']
                    for job in jobs_chunk:
                        command.append(os.path.join(run, job))
                        n_moved += 1
                    command.append(out_dir_run)
                    # Executing the move command
                    subprocess.call(command)
                    print('        Moved {0:d} jobs to {1:s}'.format(n_moved, out_dir_run))

            print('- - - - - - - -')
    # Cleaning up folders with failed jobs
    if args.clean:
        to_clean = ['out.log', 'err.txt', 'out.txt', 'log.txt', 'core.*', 'exitStatus.txt']
        failed_dir = os.path.join(args.path, 'FAILED')

        for clpath in to_clean:
            for clfile in glob.glob(failed_dir+"/Run*/job*/"+clpath):
                os.remove(clfile)

        print('### Cleaned job output files from failed jobs in: {0:s}'.format(failed_dir))

    # Cleaning up empty run folders after separate
    if args.separate:
        runDirs = glob.glob(args.path+"/Run*")
        for dir in runDirs:
            if not os.listdir(dir):
                os.rmdir(dir)
 



