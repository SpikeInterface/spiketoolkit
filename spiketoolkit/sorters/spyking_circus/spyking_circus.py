import sys

import spikeinterface as si

import os
from shutil import copyfile
import subprocess, shlex
import h5py
from mountainlab_pytools import mdaio
import numpy as np

# Installation notes for ubuntu:
# sudo apt install libopenmpi-dev
# pip install spyking-circus

def spyking_circus(*,
    recording,
    tmpdir, # Temporary working directory
    detect_sign, # Polarity of the spikes, -1, 0, or 1
    adjacency_radius, # Channel neighborhood adjacency radius corresponding to geom file
    spike_thresh=6, # Threshold for detection
    template_width_ms=3, # Spyking circus parameter
    whitening_max_elts=1000, # I believe it relates to subsampling and affects compute time
    clustering_max_elts=10000 # I believe it relates to subsampling and affects compute time
):      
    source_dir=os.path.dirname(os.path.realpath(__file__))
    
    dataset_dir=tmpdir+'/sc_dataset'
    si.MdaRecordingExtractor.writeRecording(recording_extractor=recording,save_path=dataset_dir)
        
    samplerate=recording.getSamplingFrequency()

    print('Reading timeseries header...')
    HH=mdaio.readmda_header(dataset_dir+'/raw.mda')
    num_channels=HH.dims[0]
    num_timepoints=HH.dims[1]
    duration_minutes=num_timepoints/samplerate/60
    print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(num_channels,num_timepoints,duration_minutes))
        
    print('Creating .prb file...')
    prb_text=_read_text_file(source_dir+'/template.prb')
    prb_text=prb_text.replace('$num_channels$','{}'.format(num_channels))
    prb_text=prb_text.replace('$radius$','{}'.format(adjacency_radius))
    geom=np.genfromtxt(dataset_dir+'/geom.csv', delimiter=',')
    geom_str='{\n'
    for m in range(geom.shape[0]):
        geom_str+='  {}: [{},{}],\n'.format(m,geom[m,0],geom[m,1]) # todo: handle 3d geom
    geom_str+='}'
    prb_text=prb_text.replace('$geometry$','{}'.format(geom_str))
    _write_text_file(dataset_dir+'/geometry.prb',prb_text)
        
    print('Creating .params file...')
    txt=_read_text_file(source_dir+'/template.params')
    txt=txt.replace('$header_size$','{}'.format(HH.header_size))
    txt=txt.replace('$prb_file$',dataset_dir+'/geometry.prb')
    txt=txt.replace('$dtype$',HH.dt)
    txt=txt.replace('$num_channels$','{}'.format(num_channels))
    txt=txt.replace('$samplerate$','{}'.format(samplerate))
    txt=txt.replace('$template_width_ms$','{}'.format(template_width_ms))
    txt=txt.replace('$spike_thresh$','{}'.format(spike_thresh))
    if detect_sign>0:
        peaks_str='positive'
    elif detect_sign<0:
        peaks_str='negative'
    else:
        peaks_str='both'
    txt=txt.replace('$peaks$',peaks_str)
    txt=txt.replace('$whitening_max_elts$','{}'.format(whitening_max_elts))
    txt=txt.replace('$clustering_max_elts$','{}'.format(clustering_max_elts))
    _write_text_file(dataset_dir+'/raw.params',txt)
        
    print('Running spyking circus...')
    #num_threads=np.maximum(1,int(os.cpu_count()/2))
    num_threads=np.maximum(1,int(os.cpu_count()/2))
    #num_threads=1 # for some reason, using more than 1 thread causes an error
    cmd='spyking-circus {} -c {} '.format(dataset_dir+'/raw.mda',num_threads)
    print(cmd)
    retcode=_run_command_and_print_output(cmd)

    if retcode != 0:
        raise Exception('Spyking circus returned a non-zero exit code')

    result_fname=dataset_dir+'/raw/raw.result.hdf5'
    if not os.path.exists(result_fname):
        raise Exception('Result file does not exist: '+result_fname)
    
    firings=sc_results_to_firings(result_fname)
    sorting=si.NumpySortingExtractor()
    sorting.setTimesLabels(firings[1,:],firings[2,:])
    return sorting

def sc_results_to_firings(hdf5_path):
    X=h5py.File(hdf5_path,'r');
    spiketimes=X.get('spiketimes')
    names=list(spiketimes.keys())
    clusters=[]
    for j in range(len(names)):
        times0=spiketimes.get(names[j])
        clusters.append(dict(
            k=j+1,
            times=times0
        ))
    times_list=[]
    labels_list=[]
    for cluster in clusters:
        times0=cluster['times']
        k=cluster['k']
        times_list.append(times0)
        labels_list.append(np.ones(times0.shape)*k)
    times=np.concatenate(times_list)
    labels=np.concatenate(labels_list)
    
    sort_inds=np.argsort(times)
    times=times[sort_inds]
    labels=labels[sort_inds]
    
    L=len(times)
    firings=np.zeros((3,L))
    firings[1,:]=times
    firings[2,:]=labels
    return firings

def _read_text_file(fname):
    with open(fname) as f:
        return f.read()
    
def _write_text_file(fname,str):
    with open(fname,'w') as f:
        f.write(str)
        
def _run_command_and_print_output(command):
    with subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        while True:
            output_stdout= process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc
