#!/bin/bash

old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	exit
fi

rm /scratch/pulsar/STOP
while [[ 1 ]] ; do
    # The below cleanup is handled now by the python code

    # Just keep trying... but if it crashes wait 30s
    python /scratch/pulsar/monitor_code/make_live_plots.py --backends ROACH,COBRA2 --stopflag /scratch/pulsar/STOP
    if [[ -e /scratch/pulsar/STOP ]] ; then
        break
    fi
    sleep 10
done
