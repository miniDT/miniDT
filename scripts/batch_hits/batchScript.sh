#!/bin/bash
# cat /proc/cpuinfo
# cat /proc/meminfo

SUBDIR=$1
WORKDIR=$1/$2
echo "Starting job at: $WORKDIR"

# Configure the environment
cd $SUBDIR/../../
source setenv.sh
cd $WORKDIR

cmd="python $SUBDIR/offline_analysis.py -tre `cat inputFiles.txt`"
echo "Executing command: ${cmd}"
${cmd} &> log.txt
exitStatus=$?

echo 'Finished running: ' `date` with exit status: $exitStatus
if [ $exitStatus -ne 0 ]; then
  echo $exitStatus > exitStatus.txt
fi
exit $exitStatus
