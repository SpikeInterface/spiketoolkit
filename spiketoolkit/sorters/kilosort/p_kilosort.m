function p_kilosort(kilosort_src, ironclust_src, temp_path, raw_fname, geom_fname, firings_out_fname, arg_fname)
% cmdstr2 = sprintf("p_ironclust('$(tempdir)','$timeseries$','$geom$','$firings_out$','$(argfile)');");

if exist(temp_path, 'dir') ~= 7
    mkdir(temp_path);
end

% prepare for kilosort execution
addpath(genpath(kilosort_src));
addpath(fullfile(ironclust_src, 'matlab'), fullfile(ironclust_src, 'mdaio'), fullfile(ironclust_src, 'npy-matlab'));    
ops = import_ksort_(raw_fname, geom_fname, arg_fname, temp_path);

% Run kilosort
t1=tic;
fprintf('Running kilosort on %s\n', raw_fname);
[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)
try
    rez = merge_posthoc2(rez);
catch
    fprintf(2, 'merge_posthoc2 error. Reporting pre-merge result\n'); 
end
fprintf('\n\ttook %0.1fs\n', toc(t1));

% Export kilosort
mr_out = export_ksort_(rez, firings_out_fname);

fprintf('Clustering result wrote to %s\n', firings_out_fname);

end %func


%--------------------------------------------------------------------------
function mr_out = export_ksort_(rez, firings_out_fname)

mr_out = zeros(size(rez.st3,1), 3, 'double'); 
mr_out(:,2) = rez.st3(:,1); %time
mr_out(:,3) = rez.st3(:,2); %cluster
writemda(mr_out', firings_out_fname, 'float32');
end %func


%--------------------------------------------------------------------------
function ops = import_ksort_(raw_fname, geom_fname, arg_fname, fpath)
% fpath: output path
S_txt = irc('call', 'meta2struct', {arg_fname});
[spkTh, useGPU] = deal(-abs(S_txt.detect_threshold), 1);

% convert to binary file (int16)
fbinary = strrep(raw_fname, '.mda', '.bin');
[Nchannels, ~] = mda2bin_(raw_fname, fbinary, S_txt.detect_sign);

% create a probe file
mrXY_site = csvread(geom_fname);
vcFile_chanMap = fullfile(fpath, 'chanMap.mat');
createChannelMapFile_(vcFile_chanMap, Nchannels, S_txt.samplerate, mrXY_site(:,1), mrXY_site(:,2));

ops = config_eMouse_(fpath, fbinary, vcFile_chanMap, spkTh, useGPU); %obtain ops

end %func


%--------------------------------------------------------------------------
function S = makeStruct_(varargin)
%MAKESTRUCT all the inputs must be a variable. 
%don't pass function of variables. ie: abs(X)
%instead create a var AbsX an dpass that name
S = struct();
for i=1:nargin, S.(inputname(i)) =  varargin{i}; end
end %func


%--------------------------------------------------------------------------
function S_chanMap = createChannelMapFile_(vcFile_channelMap, Nchannels, fs, xcoords, ycoords, kcoords)
if nargin<6, kcoords = []; end

connected = true(Nchannels, 1);
chanMap   = 1:Nchannels;
chanMap0ind = chanMap - 1;

xcoords   = xcoords(:);
ycoords   = ycoords(:);

if isempty(kcoords)
    kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)
end

S_chanMap = makeStruct_(chanMap, connected, xcoords, ycoords, kcoords, chanMap0ind, fs);
save(vcFile_channelMap, '-struct', 'S_chanMap')
end %func


%--------------------------------------------------------------------------
% convert mda to int16 binary format, flip polarity if detect sign is
% positive
function [nChans, nSamples] = mda2bin_(raw_fname, fbinary, detect_sign)

mr = readmda(raw_fname);
[nChans, nSamples] = size(mr);
if detect_sign > 0, mr = -mr; end % force negative detection
fid = fopen(fbinary, 'w');
fwrite(fid, mr, 'int16');
fclose(fid);
end %func


%--------------------------------------------------------------------------
function ops = config_eMouse_(fpath, fbinary, vcFile_chanMap, spkTh, useGPU)

ops = struct();
ops.GPU                 = useGPU; % whether to run this code on an Nvidia GPU (much faster, mexGPUall first)		
ops.parfor              = 0; % whether to use parfor to accelerate some parts of the algorithm		
ops.verbose             = 1; % whether to print command line progress		
ops.showfigures         = 1; % whether to plot figures during optimization		
		
ops.datatype            = 'bin';  % binary ('dat', 'bin') or 'openEphys'		
ops.fbinary             = fbinary; % will be created for 'openEphys'		
ops.fproc               = fullfile(fpath, 'temp_wh.dat'); % residual from RAM of preprocessed data		
ops.root                = fpath; % 'openEphys' only: where raw files are		
% define the channel map as a filename (string) or simply an array		
ops.chanMap             = vcFile_chanMap; % make this file using createChannelMapFile.m		
% ops.chanMap = 1:ops.Nchan; % treated as linear probe if unavailable chanMap file		

S_prb = load(ops.chanMap);
nChans = numel(S_prb.chanMap);

ops.Nfilt               = nChans*8;  % number of clusters to use (2-4 times more than Nchan, should be a multiple of 32)     		
ops.nNeighPC            = min(12,nChans); % visualization only (Phy): number of channnels to mask the PCs, leave empty to skip (12)		
ops.nNeigh              = 16; % visualization only (Phy): number of neighboring templates to retain projections of (16)		
		
% options for channel whitening		
ops.whitening           = 'full'; % type of whitening (default 'full', for 'noSpikes' set options for spike detection below)		
ops.nSkipCov            = 1; % compute whitening matrix from every N-th batch (1)		
ops.whiteningRange      = 32; % how many channels to whiten together (Inf for whole probe whitening, should be fine if Nchan<=32)		
		
ops.criterionNoiseChannels = 0.2; % fraction of "noise" templates allowed to span all channel groups (see createChannelMapFile for more info). 		

% other options for controlling the model and optimization		
ops.Nrank               = 3;    % matrix rank of spike template model (3)		
ops.nfullpasses         = 6;    % number of complete passes through data during optimization (6)		
ops.maxFR               = 20000;  % maximum number of spikes to extract per batch (20000)		
ops.fshigh              = 200;   % frequency for high pass filtering		
% ops.fslow             = 2000;   % frequency for low pass filtering (optional)
ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection		
ops.scaleproc           = 200;   % int16 scaling of whitened data		
ops.NT                  = 128*1024+ ops.ntbuff;% this is the batch size (try decreasing if out of memory) 		
% for GPU should be multiple of 32 + ntbuff		
		
% the following options can improve/deteriorate results. 		
% when multiple values are provided for an option, the first two are beginning and ending anneal values, 		
% the third is the value used in the final pass. 		
ops.Th               = [4 10 10];    % threshold for detecting spikes on template-filtered data ([6 12 12])		
ops.lam              = [5 5 5];   % large means amplitudes are forced around the mean ([10 30 30])		
ops.nannealpasses    = 4;            % should be less than nfullpasses (4)		
ops.momentum         = 1./[20 400];  % start with high momentum and anneal (1./[20 1000])		
ops.shuffle_clusters = 1;            % allow merges and splits during optimization (1)		
ops.mergeT           = .1;           % upper threshold for merging (.1)		
ops.splitT           = .1;           % lower threshold for splitting (.1)		
		
% options for initializing spikes from data		
ops.initialize      = 'fromData';    %'fromData' or 'no'		
ops.spkTh           = spkTh;      % spike threshold in standard deviations (4)		
ops.loc_range       = [3  1];  % ranges to detect peaks; plus/minus in time and channel ([3 1])		
ops.long_range      = [30  6]; % ranges to detect isolated peaks ([30 6])		
ops.maskMaxChannels = 5;       % how many channels to mask up/down ([5])		
ops.crit            = .65;     % upper criterion for discarding spike repeates (0.65)		
ops.nFiltMax        = 10000;   % maximum "unique" spikes to consider (10000)		
		
% load predefined principal components (visualization only (Phy): used for features)		
dd                  = load('PCspikes2.mat'); % you might want to recompute this from your own data		
ops.wPCA            = dd.Wi(:,1:7);   % PCs 		
		
% options for posthoc merges (under construction)		
ops.fracse  = 0.1; % binning step along discriminant axis for posthoc merges (in units of sd)		
ops.epu     = Inf;		
		
ops.ForceMaxRAMforDat   = 20e9; % maximum RAM the algorithm will try to use; on Windows it will autodetect.

end % func
