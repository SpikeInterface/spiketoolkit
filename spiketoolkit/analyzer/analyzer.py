import numpy as np
import spiketoolkit as st
from sklearn.decomposition import PCA

from spikeinterface.RecordingExtractor import RecordingExtractor
from spikeinterface.SortingExtractor import SortingExtractor

class Analyzer(object):
    '''A class that handles RecordingExtractor and SortingExtractor objects and performs
    standardized analysis and evaluation on spike sorting sorting.

    Attributes:
        recording_extractor (RecordingExtractor)
        sorting_extractor (RecordingExtractor)
    '''
    def __init__(self, recording_extractor, sorting_extractor):
        '''No need to initalize the parent class with any parameters (unless we
        agree on a standard attribute every spike sorter needs)
        '''
        # to perform comparisons between spike sorters

        if isinstance(recording_extractor, RecordingExtractor):
            self.recording_extractor = recording_extractor
        else:
            raise AttributeError('Recording extractor argument should be an RecordingExtractor object')
        if isinstance(sorting_extractor, SortingExtractor):
            self.sorting_extractor = sorting_extractor
        else:
            raise AttributeError('Sorting extractor argument should be an SortingExtractor object')

        self._waveforms = [None] * len(self.sorting_extractor.getUnitIds())
        self._templates = [None] * len(self.sorting_extractor.getUnitIds())
        self._pcascores = [None] * len(self.sorting_extractor.getUnitIds())
        self._maxchannels = [None] * len(self.sorting_extractor.getUnitIds())
        self._params = {}


    def getRecordingExtractor(self):
        '''This function returns the recording extractor and allows tu call its methods

        Returns
        ----------
        recording_extractor (RecordingExctractor)
        '''
        return self.recording_extractor

    def getSortingExtractor(self):
        '''This function returns the sorting extractor and allows tu call its methods

        Returns
        ----------
        sorting_extractor (SortingExctractor)
        '''
        return self.sorting_extractor

    def getUnitWaveforms(self, unit_ids=None, start_frame=None, end_frame=None,
                         ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False, bandpass=[300, 6000]):
        '''This function returns the spike waveforms from the specified unit_ids from t_start and t_stop
        in the form of a numpy array of spike waveforms.

        Parameters
        ----------
        unit_ids: (int or list)
            The unit id or list of unit ids to extract waveforms from
        start_frame: (int)
            The starting frame to extract waveforms
        end_frame: (int)
            The ending frame to extract waveforms
        ms_before: float
            Time in ms to cut out waveform before the peak
        ms_after: float
            Time in ms to cut out waveform after the peak

        Returns
        -------
        waveforms: np.array
            A list of 3D arrays that contain all waveforms between start and end_frame
            Dimensions of each element are: (numm_spikes x num_channels x num_spike_frames)

        '''
        if isinstance(unit_ids, (int, np.integer)):
            unit_ids = [unit_ids]
        elif unit_ids is None:
            unit_ids = self.sorting_extractor.getUnitIds()
        elif not isinstance(unit_ids, (list, np.ndarray)):
            raise Exception("unit_ids is not a valid in valid")

        params = {'start_frame': start_frame, 'end_frame': end_frame, 'ms_before': ms_before, 'ms_after': ms_after,
                  'max_num_waveforms': max_num_waveforms}

        if self._params is None:
            self._params = params

        waveform_list = []
        for i, idx in enumerate(unit_ids):
            unit_ind = np.where(np.array(self.sorting_extractor.getUnitIds()) == idx)[0]
            if len(unit_ind) == 0:
                raise Exception("unit_ids is not in valid")
            else:
                unit_ind = unit_ind[0]
                if self._waveforms[unit_ind] is None or self._params != params:
                    self._params = params
                    if not filter:
                        recordings = self.recording_extractor.getTraces(start_frame, end_frame)
                    else:
                        recordings = st.filters.bandpass_filter(recording=self.recording_extractor, freq_min=bandpass[0],
                                                                freq_max=bandpass[1]).getTraces(start_frame, end_frame)
                    fs = self.recording_extractor.getSamplingFrequency()
                    times = np.arange(recordings.shape[1])
                    spike_times = self.sorting_extractor.getUnitSpikeTrain(unit_ind, start_frame, end_frame)

                    n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

                    num_channels, num_frames = recordings.shape
                    num_spike_frames = np.sum(n_pad)

                    waveforms = np.zeros((len(spike_times), num_channels, num_spike_frames))
                    print('Waveform ' + str(i+1) + '/' + str(len(unit_ids))
                          + ' - Number of waveforms: ', len(spike_times))

                    for t_i, t in enumerate(spike_times):
                        idx = np.where(times > t)[0]
                        if len(idx) != 0:
                            idx = idx[0]
                            # find single waveforms crossing thresholds
                            if idx - n_pad[0] > 0 and idx + n_pad[1] < num_frames:
                                t_spike = times[idx - n_pad[0]:idx + n_pad[1]]
                                wf = recordings[:, idx - n_pad[0]:idx + n_pad[1]]
                            elif idx - n_pad[0] < 0:
                                t_spike = times[:idx + n_pad[1]]
                                t_spike = np.pad(t_spike, (np.abs(idx - n_pad[0]), 0), 'constant')
                                wf = recordings[:, :idx + n_pad[1]]
                                wf = np.pad(wf, ((0, 0), (np.abs(idx - n_pad[0]), 0)), 'constant')
                            elif idx + n_pad[1] > num_frames:
                                t_spike = times[idx - n_pad[0]:]
                                t_spike = np.pad(t_spike, (0, idx + n_pad[1] - num_frames), 'constant')
                                wf = recordings[:, idx - n_pad[0]:]
                                wf = np.pad(wf, ((0, 0), (0, idx + n_pad[1] - num_frames)), 'constant')
                            waveforms[t_i] = wf
                    self._waveforms[unit_ind] = waveforms
                    waveform_list.append(waveforms)
                else:
                    waveform_list.append(self._waveforms[unit_ind])

        if len(waveform_list) == 1:
            return waveform_list[0]
        else:
            return waveform_list


    def getUnitTemplate(self, unit_ids=None, **kwargs):
        '''

        Parameters
        ----------
        unit_ids
        start_frame
        end_frame

        Returns
        -------

        '''
        if isinstance(unit_ids, int):
            unit_ids = [unit_ids]
        elif unit_ids is None:
            unit_ids = self.sorting_extractor.getUnitIds()
        elif not isinstance(unit_ids, (list, np.ndarray)):
            raise Exception("unit_ids is not a valid in valid")

        template_list = []
        for idx in unit_ids:
            unit_ind = np.where(np.array(self.sorting_extractor.getUnitIds()) == idx)[0]
            if len(unit_ind) == 0:
                raise Exception("unit_ids is not in valid")
            else:
                unit_ind = unit_ind[0]
            if self._templates[unit_ind] is None or self._params != kwargs:
                self._params.update(kwargs)
                if self._waveforms[unit_ind] is None:
                    self.getUnitWaveforms(unit_ind, **kwargs)
                template = np.mean(self._waveforms[unit_ind], axis = 0)
                self._templates[unit_ind] = template
                template_list.append(template)
            else:
                template = self._templates[unit_ind]
                template_list.append(template)

        if len(template_list) == 1:
            return template_list[0]
        else:
            return template_list


    def getUnitMaxChannel(self, unit_ids=None, **kwargs):
        '''

        Parameters
        ----------
        unit_ids

        Returns
        -------

        '''
        if isinstance(unit_ids, int):
            unit_ids = [unit_ids]
        elif unit_ids is None:
            unit_ids = self.sorting_extractor.getUnitIds()
        elif not isinstance(unit_ids, (list, np.ndarray)):
            raise Exception("unit_ids is not a valid in valid")

        max_list = []
        for idx in unit_ids:
            unit_ind = np.where(np.array(self.sorting_extractor.getUnitIds()) == idx)[0]
            if len(unit_ind) == 0:
                raise Exception("unit_ids is not in valid")
            else:
                unit_ind = unit_ind[0]

            if self._maxchannels[unit_ind] is None or self._params != kwargs:
                self._params.update(kwargs)
                if self._templates[unit_ind] is None:
                    self.getUnitTemplate(unit_ids, **kwargs)
                max_channel = np.unravel_index(np.argmax(np.abs(self._templates[unit_ind])),
                                            self._templates[unit_ind].shape)[0]
                self._maxchannels[unit_ind] = max_channel
                max_list.append(max_channel)
            else:
                max_channel = self._maxchannels[unit_ind]
                max_list.append(max_channel)

        if len(max_list) == 1:
            return max_list[0]
        else:
            return max_list


    def computePCAscores(self, n_comp=10):
        '''

        Parameters
        ----------
        n_comp

        Returns
        -------

        '''
        # concatenate all waveforms
        all_waveforms = np.array([])
        nspikes = []
        for i_w, wf in enumerate(self._waveforms):
            if wf is None:
                self.getUnitWaveforms(self.sorting_extractor.getUnitIds()[i_w])
            nspikes.append(len(wf))
            wf_reshaped = wf.reshape((wf.shape[0], wf.shape[1]*wf.shape[2]))
            if i_w == 0:
                all_waveforms = wf_reshaped
            else:
                np.concatenate((all_waveforms, wf_reshaped))
        print(all_waveforms.shape)
        print("Fitting PCA of %d dimensions" % n_comp)
        pca = PCA(n_components=n_comp, whiten=True)
        pca.fit_transform(all_waveforms.T)
        scores = pca.transform(all_waveforms.T)
        var = pca.explained_variance_ratio_

        init = 0
        pca_scores = []
        for i_n, nsp in enumerate(nspikes):
            self._pcascores[i_n] = scores[init : init + nsp]
            init = nsp + 1
            pca_scores.append(self._pcascores[i_n])

        return pca_scores


