#!/bin/bash


old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
echo "Stopping existing monitor..."
	/scratch/pulsar/monitor_code/stop_monitor.sh
fi

old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	echo "Alas... the old monitor code is still running."
	echo "Choosing not to run a new one"
	echo "Please try to terminate the existing 'make_live_plots.py' script"
	exit
fi


nohup /scratch/pulsar/monitor_code/live_backend_monitor.sh > /dev/null &
echo "Stating..."
sleep 1
ps -f $(pgrep -f make_live_plots.py)
