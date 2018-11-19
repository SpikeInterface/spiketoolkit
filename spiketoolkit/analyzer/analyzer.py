import numpy as np
import spiketoolkit as st
from sklearn.decomposition import PCA

from spikeextractors.RecordingExtractor import RecordingExtractor
from spikeextractors.SortingExtractor import SortingExtractor

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

        self._waveforms = {}
        self._templates = {}
        self._pcascores = {}
        self._maxchannels = {}
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
                         ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False, bandpass=[300, 6000],
                         verbose=True):
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
        for i, unit_ind in enumerate(unit_ids):
            if unit_ind not in self.sorting_extractor.getUnitIds():
                raise Exception("unit_ids is not in valid")
            if unit_ind not in self._waveforms.keys() or self._params != params:
                self._params = params
                if not filter:
                    recordings = self.recording_extractor.getTraces(start_frame, end_frame)
                else:
                    recordings = st.filters.bandpass_filter(recording=self.recording_extractor, freq_min=bandpass[0],
                                                            freq_max=bandpass[1]).getTraces(start_frame, end_frame)
                fs = self.recording_extractor.getSamplingFrequency()
                times = np.arange(recordings.shape[1])
                spike_times = self.sorting_extractor.getUnitSpikeTrain(unit_ind, start_frame, end_frame)

                if len(spike_times) > max_num_waveforms:
                    spike_times = spike_times[np.random.permutation(len(spike_times))[:max_num_waveforms]]

                n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

                num_channels, num_frames = recordings.shape
                num_spike_frames = np.sum(n_pad)

                waveforms = np.zeros((len(spike_times), num_channels, num_spike_frames))
                if verbose:
                    print('Waveform ' + str(i+1) + '/' + str(len(unit_ids))
                          + ' - Number of waveforms: ', len(spike_times))

                waveforms = self._get_random_spike_waveforms(unit=unit_ind,
                                                             max_num=max_num_waveforms,
                                                             snippet_len=n_pad)
                waveforms = waveforms.swapaxes(0,2)
                waveforms = waveforms.swapaxes(1,2)

                # for t_i, t in enumerate(spike_times):
                #     idx = np.where(times > t)[0]
                #     if len(idx) != 0:
                #         idx = idx[0]
                #         # find single waveforms crossing thresholds
                #         if idx - n_pad[0] > 0 and idx + n_pad[1] < num_frames:
                #             wf = recordings[:, idx - n_pad[0]:idx + n_pad[1]]
                #         elif idx - n_pad[0] < 0:
                #             wf = recordings[:, :idx + n_pad[1]]
                #             wf = np.pad(wf, ((0, 0), (np.abs(idx - n_pad[0]), 0)), 'constant')
                #         elif idx + n_pad[1] > num_frames:
                #             wf = recordings[:, idx - n_pad[0]:]
                #             wf = np.pad(wf, ((0, 0), (0, idx + n_pad[1] - num_frames)), 'constant')
                #         waveforms[t_i] = wf
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
        if isinstance(unit_ids, (int, np.integer)):
            unit_ids = [unit_ids]
        elif unit_ids is None:
            unit_ids = self.sorting_extractor.getUnitIds()
        elif not isinstance(unit_ids, (list, np.ndarray)):
            raise Exception("unit_ids is not a valid in valid")

        template_list = []
        for i, unit_ind in enumerate(unit_ids):
            if unit_ind not in self.sorting_extractor.getUnitIds():
                raise Exception("unit_ids is not in valid")
            if unit_ind not in self._templates.keys() or self._params != kwargs:
                self._params.update(kwargs)
                if unit_ind not in self._waveforms.keys():
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
        if isinstance(unit_ids, (int, np.integer)):
            unit_ids = [unit_ids]
        elif unit_ids is None:
            unit_ids = self.sorting_extractor.getUnitIds()
        elif not isinstance(unit_ids, (list, np.ndarray)):
            raise Exception("unit_ids is not a valid in valid")

        max_list = []
        for i, unit_ind in enumerate(unit_ids):
            if unit_ind not in self.sorting_extractor.getUnitIds():
                raise Exception("unit_ids is not in valid")
            if unit_ind not in self._maxchannels.keys() is None or self._params != kwargs:
                self._params.update(kwargs)
                if unit_ind not in self._templates.keys():
                    self.getUnitTemplate(unit_ind, **kwargs)
                max_channel = np.unravel_index(np.argmax(np.abs(self._templates[i])),
                                            self._templates[i].shape)[0]
                self._maxchannels[unit_ind] = max_channel
                max_list.append(max_channel)
            else:
                max_channel = self._maxchannels[unit_ind]
                max_list.append(max_channel)

        if len(max_list) == 1:
            return max_list[0]
        else:
            return max_list


    def computePCAscores(self, n_comp=3, elec=False, max_num_waveforms=np.inf):
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
        for i_w, wf in enumerate(self.getUnitWaveforms()):
            if wf is None:
                wf = self.getUnitWaveforms(self.sorting_extractor.getUnitIds()[i_w], verbose=True)
            if elec:
                wf_reshaped = wf.reshape((wf.shape[0]*wf.shape[1], wf.shape[2]))
                nspikes.append(len(wf)*self.recording_extractor.getNumChannels())
            else:
                wf_reshaped = wf.reshape((wf.shape[0], wf.shape[1] * wf.shape[2]))
                nspikes.append(len(wf))
            if i_w == 0:
                all_waveforms = wf_reshaped
            else:
                all_waveforms = np.concatenate((all_waveforms, wf_reshaped))
        print("Fitting PCA of %d dimensions on %d waveforms" % (n_comp, len(all_waveforms)))

        pca = PCA(n_components=n_comp, whiten=True)
        pca.fit_transform(all_waveforms)
        scores = pca.transform(all_waveforms)

        init = 0
        pca_scores = []
        for i_n, nsp in enumerate(nspikes):
            pcascores = scores[init : init + nsp, :]
            init = nsp + 1
            if elec:
                pca_scores.append(pcascores.reshape(nsp//self.recording_extractor.getNumChannels(),
                                                    self.recording_extractor.getNumChannels(), n_comp))
            else:
                pca_scores.append(pcascores)

        return np.array(pca_scores)


    def _get_random_spike_waveforms(self, *, unit, max_num, snippet_len, channels=None):
        st=self.sorting_extractor.getUnitSpikeTrain(unit_id=unit)
        num_events=len(st)
        if num_events>max_num:
            event_indices=np.random.choice(range(num_events),size=max_num,replace=False)
        else:
            event_indices=range(num_events)

        spikes=self.recording_extractor.getSnippets(reference_frames=st[event_indices].astype(int),
                                                    snippet_len=snippet_len, channel_ids=channels)
        spikes=np.dstack(tuple(spikes))
        return spikes

