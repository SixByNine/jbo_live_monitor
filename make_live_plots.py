#!/usr/bin/env python

import glob
import os
import subprocess
import json
import datetime
import time

import argparse
import logging

import yaml

import matplotlib
matplotlib.use("agg")
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import seaborn

plt.rcParams.update({'font.size': 20, 'font.family':'serif'})

def iqrmask(phsfreq):
    stat=np.std(phsfreq,axis=1)
    stat -= np.median(stat)
    q25, q75 = np.percentile(stat, 25), np.percentile(stat, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    zap = np.logical_or(stat > upper,stat < lower)
    return zap


def combine_freq(ars):
    fappend = psrchive.FrequencyAppend()
    outar = ars[0]
    fappend.init(outar)
    for ar in ars[1:]:
        fappend.append(outar,ar)
    return outar


def combine_time(ars):
    tappend = psrchive.TimeAppend()
    outar = ars[0]
    tappend.init(outar)
    for ar in ars[1:]:
        tappend.append(outar,ar)
    return outar



def loop(backend=None):
    if backend=="ROACH":
        workdir="/scratch/pulsar/ROACH_MONITOR"
        nodeshellmask="compute*/" # ROACH compute nodes
        isROACH=True
        nodenames=[f"compute-0-{inode}" for inode in range(33)]
        outpath="pulsar@liveweb:./live_roach"
    elif backend=="COBRA2":
        workdir="/scratch/pulsar/COBRA2_MONITOR"
        nodeshellmask="" # COBRA2 doesn't have any sub-nodes.
        isROACH=False
        nodenames=["cobra2"]
        outpath="pulsar@liveweb:./live_cobra2"

    logging.info(f"Working in {workdir}")
    os.chdir(workdir)
    # clear old cache files...
    subprocess.call(["find","plots","cache","-type","f","-mmin","+60","-delete"])
    # clear old node state files...
    subprocess.call(["find","state","-type","f","-mmin","+5","-delete"])

    arlist={}
    arsearch = f"indata/"
    logging.debug(f"Looking archives in {workdir}/{arsearch}")
    for root, _, files in os.walk("indata/"):
        for file in files:
            if file.endswith(".ar"):
                arname=file
                if arname not in arlist:
                    logging.debug(f"Found {os.path.join(root,file)}")
                    arlist[arname]=os.path.join(root,file)

    logging.debug(f"Found {len(arlist)} archives")

    valid_arnames=arlist.keys()
    logging.debug(f"Valid arnames {valid_arnames}")

    lastar=sorted(valid_arnames)[-1]
    last_observation = os.path.basename(os.path.dirname(arlist[lastar]))
    logging.info(f"Most recent observation= {last_observation}")


    

    time.sleep(1) # Try to give a chance in case data is being copied.




    arlist={}
    ar_count=0


    for root, _, files in os.walk("indata/"):
        if root.endswith(f"{last_observation}"):
            logging.debug(f"Looking archives in {root}")
            for file in files:
                if file.endswith(".ar"):
                    arname=file
                    if arname in valid_arnames:
                        if arname not in arlist:
                            arlist[arname]=[]
                        logging.debug(f"Found {os.path.join(root,file)}")
                        arlist[arname].append(os.path.join(root,file))
                        ar_count += 1


    
    # for path in glob.glob(f"indata/{nodeshellmask}{last_observation}/*.ar"):
    #     arname = os.path.basename(path)
    #     if arname in valid_arnames:
    #         if arname not in arlist:
    #             arlist[arname]=[]
    #         arlist[arname].append(path)
    #         ar_count += 1


    try:
        logging.debug(f"try load cache {last_observation}")
        cache=np.load(f"cache/{last_observation}_cache.npz")
        phase_freq=cache['phase_freq']
        ars_cached=cache['ars_cached'].tolist()
        subint_counts=cache['subint_counts'].tolist()
        phase_time=cache['phase_time'].tolist()
        real_freq_table=cache['real_freq_table']
        freq_counts=cache['freq_counts']
        prev_freq_counts=cache['prev_freq_counts']
        prev_phase_freq=cache['prev_phase_freq']
        logging.debug(f"got cache {last_observation}")
        badcount=0
        for i, arname in enumerate(sorted(arlist.keys())[:len(subint_counts)]):
            if subint_counts[i] < len(arlist[arname]):
                if i == len(subint_counts)-1:
                    logging.info(f"Reverting last subint {subint_counts[i]} =/= {len(arlist[arname])}")
                    # The last subint was missing some stuff. Avoid rebuilding everything by just reverting by one
                    subint_counts = subint_counts[:-1]
                    ars_cached = ars_cached[:-1]
                    phase_time = phase_time[:-1]
                    phase_freq = prev_phase_freq
                    freq_counts = prev_freq_counts
                    prev_phase_freq = None
                else:
                    # The new data is not simply the last subint - this means we need to remake everything
                    #logging.info(f"Don't bother trying to add in very old data...")
                    badcount += len(arlist[arname])-subint_counts[i]
        if badcount >= len(nodenames):
            raise Exception("More data has appeared... need to remake observation")
        
    except Exception as err:
        logging.warning(err)
        logging.info(f"New obs... {last_observation}")
        phase_freq=None
        phase_time=[]
        ars_cached=[]
        subint_counts=[]



    


    new_data=False
    num_archives_read=0
    number_of_available_ar=0
    logging.debug(sorted(arlist.keys())[:-1])
    for arname in sorted(arlist.keys())[:-1]:
        number_of_available_ar+=len(arlist[arname])
        if arname not in ars_cached:
            
            new_data=True
            logging.debug(f"{arname} {len(arlist[arname])}")
            #TODO: check cache
            ars=[]
            for path in arlist[arname]:
                logging.debug(path)
                num_archives_read+=1
                ars.append(psrchive.Archive_load(path))
            subint_counts.append(len(arlist[arname])) # How many subbands we read.

            combined = combine_freq(ars)
            if isROACH:
                combined.execute('edit freq=1520')
            combined.dedisperse()
            combined.pscrunch()
            combined.tscrunch()
            combined.remove_baseline()
            dat = combined.get_data()[0,0]
            
            
            
            nchan,nbin=dat.shape
            freq_table=combined.get_frequency_table()[0]
            if isROACH: # ROACH can have missing sub-bands etc...
                if phase_freq is None:
                    df = freq_table[1]-freq_table[0]
                    real_nchan = int(-400/df) # Assume 400 MHz band
                    real_freq_table = 1532.0 +(np.arange(real_nchan)-real_nchan//2 + 0.5)*df

                    phase_freq = np.zeros((real_nchan,nbin))
                    freq_counts = np.zeros(real_nchan)
            else: # Sane backend doesn't have missing sub-bands.
                if phase_freq is None:
                    real_freq_table = freq_table
                    phase_freq = np.zeros((nchan,nbin))
                    freq_counts = np.zeros(nchan)
            

            zap = iqrmask(dat)
            goodchan=np.logical_not(zap)
            
            
            mask=np.isin(real_freq_table,freq_table[goodchan])
            prev_phase_freq = phase_freq.copy()
            prev_freq_counts = freq_counts.copy()
            phase_freq[mask]+=dat[goodchan]
            freq_counts[mask]+=1

            phase_time.append(np.mean(dat[goodchan],axis=0))
            
            ars_cached.append(arname)

    liveplotid = f"{last_observation}_{number_of_available_ar}.png"
    outplot=f"plots/{liveplotid}"

    if not os.path.isfile(outplot):
        np.savez(f"cache/{last_observation}_cache",ars_cached=ars_cached,
                phase_freq=phase_freq,
                phase_time=phase_time,
                prev_phase_freq = prev_phase_freq,
                real_freq_table=real_freq_table,
                freq_counts=freq_counts,
                prev_freq_counts = prev_freq_counts,
                subint_counts=subint_counts)
        print(f"Save cache {last_observation}")

        nchan,nbin=phase_freq.shape


        pal=seaborn.color_palette("rocket", as_cmap=True)
        pal.set_bad('gray')

        z=phase_freq/freq_counts.reshape(-1,1)
        prof=np.sum(phase_time,axis=0)
        prof/=np.amax(prof)
        x=np.arange(nbin)/nbin

        logging.debug(f"{np.nanmax(z)},{np.nanmin(z)}")
        fig, ((a2,a3),(a0,a1)) = plt.subplots(2,2,gridspec_kw=dict(height_ratios=[1,5],hspace=0),figsize=(18,9),facecolor='white')

        a2.plot(x,prof,color='k')
        a2.set_xlim(0,1)
        a2.get_yaxis().set_visible(False)
        a2.get_xaxis().set_ticks([])

        a3.plot(x,prof,color='k')
        a3.set_xlim(0,1)
        a3.get_yaxis().set_visible(False)
        a3.get_xaxis().set_ticks([])
        vmax = np.nanpercentile(z, 99)
        vmin = np.nanpercentile(z, 1)

        a0.imshow(z,aspect='auto',
                extent=(0,nbin,real_freq_table[0],real_freq_table[-1]),
                origin='lower',cmap=pal,vmax=vmax,vmin=vmin,interpolation='None')
        a0.set_xlabel("Phase")
        a0.set_ylabel("Freq")

        if isROACH:
            a0_2 = a0.twinx()

            a0_2.set_yticks(np.arange(25)+1)
            offset=matplotlib.transforms.ScaledTranslation(0, -9/72, fig.dpi_scale_trans)
            for tic in a0_2.get_yticklabels():
                tic.set_fontsize(12)
                tic.set_transform(tic.get_transform() + offset)

            freq_count_x=(freq_counts/np.amax(freq_counts))*nbin/25+nbin
            a0.plot(freq_count_x,np.linspace(real_freq_table[0],real_freq_table[-1],nchan),clip_on=False,color='gray',alpha=0.5)
            a0.fill_betweenx(np.linspace(real_freq_table[0],real_freq_table[-1],nchan),
                    np.zeros(nchan)+nbin,
                    freq_count_x,
                    clip_on=False,alpha=0.2,color='gray')

            a0.autoscale(False)
            a0_2.autoscale(False)

        z=phase_time
        vmax = np.nanpercentile(z, 99)
        vmin = np.nanpercentile(z, 1)
        

        a1.imshow(z,aspect='auto',
                extent=(0,nbin,0,len(phase_time)),
                origin='lower',cmap=pal,vmax=vmax,vmin=vmin,
                interpolation='None')
        a1.set_xlabel("Phase")
        a1.set_ylabel("Subint")

        
        plt.suptitle(f"{last_observation} ({num_archives_read}/{number_of_available_ar})")
        fig.tight_layout()
        plt.savefig(outplot)

        subprocess.call(["rsync","-avP","--delete","plots/",f"{outpath}/plots/"]) # Add --delete at some point...


    nodestatelines={}
    for inode,name in enumerate(nodenames):
        try:
            with open(f"state/{name}") as f:
                nodestatelines[inode] = f.readline()
        except:
            nodestatelines[inode] = ""


    def parse_ringbuffer(state, string):
        state['ready']=False
        state['nbufs']=0
        state['clear']=0
        state['full']=0
        state['written']=0
        state['used']=0
        esplt=string.split(",")
        if len(esplt)==10:
            total, full, clear, written, _, _, _, _, _, _ = [int(e) for e in esplt]
            if total > 0:
                state['ready']=True
                state['nbufs']=total
                state['clear']=clear
                state['full']=full
                state['written']=written
                state['used']=full/total


    total_status="Unknown"
    dspsr_statuses=[]
    digifil_statuses=[]
    udpdb_statuses=[]
    nodestate={}
    nodemap={}
    for inode,name in enumerate(nodenames):
        nodestate[inode]=dict(status='Unknown', date='??')
        nodestate[inode]['ringbuffer']={'dada': dict(label="dada",key='dada', nbufs=0,used=0,full=0,clear=0,written=0,ready=False),
                'eada': dict(label="eada",key='eada', nbufs=0,used=0,full=0,clear=0,written=0,ready=False)}
        #e = nodestatelines[inode].split("||")
        # $date||$state||$load||$rb1||$rb2||$ndspsr,...||$used
        try:
            node_yaml = yaml.safe_load(nodestatelines[inode])
            if 'node' in node_yaml:
                node_yaml=node_yaml['node']
                isactive=False
                logging.debug(f"Node {name}")
                logging.debug(node_yaml)
                if 'nodepos' in node_yaml:
                    nodepos=node_yaml['nodepos']
                else:
                    nodepos=1
                nodestate[inode]['nodename']=name
                if nodepos > 0:
                    isactive=True
                    nodemap[nodepos]=inode
                lavg=node_yaml['load'].split()
                nodestate[inode]['date']=node_yaml['date']
                nodestate[inode]['status']=node_yaml['state']
                for key in  node_yaml['ringbuffers'].keys():
                    parse_ringbuffer(nodestate[inode]['ringbuffer'][key], node_yaml['ringbuffers'][key])
                
                if 'last_file' in node_yaml:
                    nodestate[inode]['last_file']=node_yaml['last_file']

                nodestate[inode]['ndspsr'] = int(node_yaml['processes']['dspsr'])
                if 'udpdb' in node_yaml['processes']:
                    nodestate[inode]['nudpdb']=int(node_yaml['processes']['udpdb'])
                elif 'cobra_dmadb' in node_yaml['processes']:
                    nodestate[inode]['nudpdb']=int(node_yaml['processes']['cobra_dmadb'])
                else:
                    nodestate[inode]['nudpdb']=0

                if 'digifil' in node_yaml['processes']:
                    nodestate[inode]['ndigifil']=int(node_yaml['processes']['digifil'])
                else:
                    nodestate[inode]['ndigifil']=0
                if 'udpdb2' in node_yaml['processes']:
                    nodestate[inode]['nudpdb2']=int(node_yaml['processes']['udpdb2'])
                else:
                    nodestate[inode]['nudpdb2']=0
                
                nodestate[inode]['active']=nodepos>0
                nodestate[inode]['nodepos']=nodepos
                if nodestate[inode]['ndspsr'] == 1 and nodestate[inode]['ringbuffer']['dada']['written']==0:
                    nodestate[inode]['dspsr'] = "waiting"
                elif nodestate[inode]['ndspsr'] == 1:
                    nodestate[inode]['dspsr'] = "running"
                elif nodestate[inode]['ndspsr'] > 1:
                    nodestate[inode]['dspsr'] = "multi"
                else:
                    nodestate[inode]['dspsr'] = "stopped"

                if nodestate[inode]['ndigifil'] == 1 and nodestate[inode]['ringbuffer']['eada']['written']==0:
                    nodestate[inode]['digifil'] = "waiting"
                elif nodestate[inode]['ndigifil'] == 1:
                    nodestate[inode]['digifil'] = "running"
                elif nodestate[inode]['ndigifil'] > 1:
                    nodestate[inode]['digifil'] = "multi"
                else:
                    if nodestate[inode]['nudpdb']==1 and nodestate[inode]['nudpdb2'] == 0:
                        nodestate[inode]['digifil'] = "disabled"
                    else:
                        nodestate[inode]['digifil'] = "stopped"
                
                if nodestate[inode]['nudpdb'] == 1:
                    nodestate[inode]['udpdb'] = "running"
                elif nodestate[inode]['nudpdb'] > 1:
                    nodestate[inode]['udpdb'] = "multi"
                else:
                    nodestate[inode]['udpdb'] = "stopped"
                if isactive:
                    dspsr_statuses.append(nodestate[inode]['dspsr'])
                    digifil_statuses.append(nodestate[inode]['digifil'])
                    udpdb_statuses.append(nodestate[inode]['udpdb'])

                nodestate[inode]['load1']=float(lavg[0])
                nodestate[inode]['load5']=float(lavg[1])
        except Exception as err:
            logging.warning(err)
            nodestate[inode]['status']="monitor_error"

    dspsr_set = set(dspsr_statuses)
    if isROACH:
        expected_nodes = 25 # Maybe this can be worked out somehow
    else:
        expected_nodes = 1
    if (len(dspsr_statuses) != expected_nodes):
        dspsr_status = "missing"
    elif len(dspsr_set) == 1:
        dspsr_status = dspsr_statuses[0]
    else:
        dspsr_status = "mixed"
    # Same for digifil
    digifil_set = set(digifil_statuses)
    if (len(digifil_statuses) != expected_nodes):
        digifil_status = "missing"
    elif len(digifil_set) == 1:
        digifil_status = digifil_statuses[0]
    else:
        if digifil_set.intersection(["running","disabled"]) == digifil_set:
            digifil_status = "running"
        elif digifil_set.intersection(["waiting","disabled"]) == digifil_set:
            digifil_status = "waiting"
        else:
            digifil_status = "mixed"
    # Same for udpdb
    udpdb_set = set(udpdb_statuses)
    if (len(udpdb_statuses) != expected_nodes):
        udpdb_status = "missing"
    elif len(udpdb_set) == 1:
        udpdb_status = udpdb_statuses[0]
    else:
        udpdb_status = "mixed"
    
    if dspsr_status == "running" and (digifil_status == "running" or digifil_status=="disabled") and udpdb_status == "running":
        total_status = "running"
    elif dspsr_status == "waiting" and digifil_status == "waiting" and udpdb_status == "running":
        total_status = "waiting"
    elif dspsr_status == "stopped" and digifil_status == "stopped" and udpdb_status == "stopped":
        total_status = "stopped"
    else:
        total_status = "mixed"

    nowdate=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state=dict(backend=backend,nodes=nodestate,liveplot=liveplotid,nodemap=nodemap,date=nowdate,
               dspsr_status=dspsr_status,digifil_status=digifil_status,udpdb_status=udpdb_status,status=total_status,
               source='unknown',requested_state='unknown',requested_state_update_time='2020-01-01T00:00:00Z',live="unknown")
    
    try:
        with open(os.path.join(workdir,"control","source.txt"),"r") as f:
            state['source'] = f.readline().strip()
            if last_observation.endswith(state['source']):
                state['live']="live"
            else:
                state['live']="notlive"
    except: 
        pass
    try:
        statefile=os.path.join(workdir,"control","state.txt")
        with open(statefile,"r") as f:
            state['requested_state'] = f.readline().strip()
            dt=os.path.getmtime(statefile)
            state['requested_state_update_time']=datetime.datetime.fromtimestamp(dt,datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        pass

    try:
        with open(os.path.join(workdir,"control","config.txt")) as f:
            state['config'] = f.readline().strip()
            if '-s ' in state['config']:
                state['live']="singlepulse"
    except:
        pass


    with open("state.json","w") as f:
        f.write(json.dumps(state))

    subprocess.call(["rsync","-avP","state.json",f"{outpath}/state.json"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make live plots')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument("--backends", type=str, help="Comma separated list of backends to run")
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    parser.add_argument("--stopflag", type=str,default="STOP", help="Stop flag file")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.backends:
        backends = [backend.upper() for backend in args.backends.split(",")]
    else:
        backends=["ROACH","COBRA2"]

    loopcount=10800*2
    while loopcount > 0:
        for backend in backends:
            loop(backend)
        if os.path.exists(args.stopflag) or args.once:
            break
        time.sleep(1)
        loopcount -= 1
    logging.info("Exiting")
