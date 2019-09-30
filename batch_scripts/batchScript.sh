#!/bin/bash
# cat /proc/cpuinfo
# cat /proc/meminfo

EXECDIR=$1/..
WORKDIR=$1/$2
echo "Starting job at: $WORKDIR"

# Configure the environment
cd $EXECDIR
source setenv.sh
cd $WORKDIR

cmd="python $EXECDIR/process_hits.py -a --hits_pos `cat inputFiles.txt`"
echo "Executing command: ${cmd}"
${cmd} &> log.txt
exitStatus=$?

echo 'Finished running: ' `date` with exit status: $exitStatus
if [ $exitStatus -ne 0 ]; then
  echo $exitStatus > exitStatus.txt
fi
exit $exitStatus
