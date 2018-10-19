% flag for GPU (must have CUDA installed)
tic
useGPU = {};

% prepare for kilosort execution
addpath(genpath('{}'));
addpath(genpath('{}'));

% set file and ouptu paths
fpath = '{}';
output = '{}';

% create channel map file
run(fullfile('kilosort_channelmap.m'));

% Run the configuration file, it builds the structure of options (ops)
run(fullfile('kilosort_config.m'))

% This part runs the normal Kilosort processing on the simulated data
[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

try
    rez = merge_posthoc2(rez);
catch
    fprintf(2, 'merge_posthoc2 error. Reporting pre-merge result\n');
end

% save python results file for Phy
mkdir(output)
rezToPhy(rez, fullfile(output));

elapsed_time = toc;
fid = fopen(fullfile(output, 'time.txt'), 'w');
fprintf(fid, '%f', elapsed_time);
fclose(fid)


