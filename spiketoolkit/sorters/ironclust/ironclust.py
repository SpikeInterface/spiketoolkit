import spikeinterface as si

import os
from mountainlab_pytools import mdaio
from spiketoolkit.sorters.tools import run_command_and_print_output

def ironclust(*,
    recording, # Recording object
    tmpdir, # Temporary working directory
    detect_sign=-1, # Polarity of the spikes, -1, 0, or 1
    adjacency_radius=-1, # Channel neighborhood adjacency radius corresponding to geom file
    detect_threshold=5, # Threshold for detection
    merge_thresh=.98, # Cluster merging threhold 0..1
    freq_min=300, # Lower frequency limit for band-pass filter
    freq_max=6000, # Upper frequency limit for band-pass filter
    pc_per_chan=3, # Number of pc per channel
    prm_template_name, # Name of the template file
    ironclust_src=None
):      
    if ironclust_src is None:
        ironclust_src=os.getenv('IRONCLUST_SRC',None)
    if not ironclust_src:
        raise Exception('You must either set the IRONCLUST_SRC environment variable, or pass the ironclust_src parameter')
    source_dir=os.path.dirname(os.path.realpath(__file__))

    dataset_dir=tmpdir+'/ironclust_dataset'
    # Generate three files in the dataset directory: raw.mda, geom.csv, params.json
    si.MdaRecordingExtractor.writeRecording(recording=recording,save_path=dataset_dir)
        
    samplerate=recording.getSamplingFrequency()

    print('Reading timeseries header...')
    HH=mdaio.readmda_header(dataset_dir+'/raw.mda')
    num_channels=HH.dims[0]
    num_timepoints=HH.dims[1]
    duration_minutes=num_timepoints/samplerate/60
    print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(num_channels,num_timepoints,duration_minutes))

    print('Creating .params file...')
    txt=''
    txt+='samplerate={}\n'.format(samplerate)
    txt+='detect_sign={}\n'.format(detect_sign)
    txt+='adjacency_radius={}\n'.format(adjacency_radius)
    txt+='detect_threshold={}\n'.format(detect_threshold)
    txt+='merge_thresh={}\n'.format(merge_thresh)
    txt+='freq_min={}\n'.format(freq_min)
    txt+='freq_max={}\n'.format(freq_max)    
    txt+='pc_per_chan={}\n'.format(pc_per_chan)
    txt+='prm_template_name={}\n'.format(prm_template_name)
    _write_text_file(dataset_dir+'/argfile.txt',txt)
        
    print('Running IronClust...')
    cmd_path = "addpath('{}', '{}/matlab', '{}/mdaio');".format(ironclust_src, ironclust_src, ironclust_src)
    #"p_ironclust('$(tempdir)','$timeseries$','$geom$','$prm$','$firings_true$','$firings_out$','$(argfile)');"
    cmd_call = "p_ironclust('{}', '{}', '{}', '', '', '{}', '{}');"\
        .format(tmpdir, dataset_dir+'/raw.mda', dataset_dir+'/geom.csv', tmpdir+'/firings.mda', dataset_dir+'/argfile.txt')
    cmd='matlab -nosplash -nodisplay -r "{} {} quit;"'.format(cmd_path, cmd_call)
    print(cmd)
    retcode=run_command_and_print_output(cmd)

    if retcode != 0:
        raise Exception('IronClust returned a non-zero exit code')

    # parse output
    result_fname=tmpdir+'/firings.mda'
    if not os.path.exists(result_fname):
        raise Exception('Result file does not exist: '+ result_fname)
    
    firings=mdaio.readmda(result_fname)
    sorting=si.NumpySortingExtractor()
    sorting.setTimesLabels(firings[1,:],firings[2,:])
    return sorting

def _read_text_file(fname):
    with open(fname) as f:
        return f.read()
    
def _write_text_file(fname,str):
    with open(fname,'w') as f:
        f.write(str)