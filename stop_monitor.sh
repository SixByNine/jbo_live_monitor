#!/bin/bash

touch /scratch/pulsar/STOP

old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	echo "wait for stop..."
sleep 10
fi
old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	echo "wait for stop..."
sleep 10
fi

old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	echo "wait for stop..."
sleep 20
fi

old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	echo "wait for stop..."
sleep 20
fi

old=$(pgrep -f make_live_plots.py)
if [[ -n "$old" ]] ; then
	pkill -f make_live_plots.py
	sleep 1
fi

