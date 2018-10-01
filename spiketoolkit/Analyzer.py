import numpy as np
import spikeinterface as si

# use InputExtractor and OutputExtractor
from spikeinterface.InputExtractor import InputExtractor
from spikeinterface.OutputExtractor import OutputExtractor

class Analyzer(object):
    '''A class that handles InputExtractor and OutputExtractor objects and performs
    standardized analysis and evaluation on spike sorting output.

    Attributes:
        input_extractor (InputExtractor)
        output_extractor (InputExtractor)
    '''
    def __init__(self, input_extractor, output_extractor):
        '''No need to initalize the parent class with any parameters (unless we
        agree on a standard attribute every spike sorter needs)
        '''
        # to perform comparisons between spike sorters
        if isinstance(input_extractor, InputExtractor):
            self.input_extractor = input_extractor
        else:
            raise AttributeError('Input extractor argument should be an InputExtractor object')
        if isinstance(output_extractor, OutputExtractor):
            self.output_extractor = output_extractor
        else:
            raise AttributeError('Output extractor argument should be an OutputExtractor object')

        self._waveforms = [None] * len(self.output_extractor.getUnitIds())
        self._templates = [None] * len(self.output_extractor.getUnitIds())
        self._pcascores = [None] * len(self.output_extractor.getUnitIds())
        self._maxchannels = [None] * len(self.output_extractor.getUnitIds())

    def inputExtractor(self):
        '''This function returns the input extractor and allows tu call its methods

        Returns
        ----------
        input_extractor (InputExctractor)
        '''
        return self.input_extractor

    def outputExtractor(self):
        '''This function returns the output extractor and allows tu call its methods

        Returns
        ----------
        output_extractor (OutputExctractor)
        '''
        return self.output_extractor

    def getUnitWaveforms(self, unit_id=None, start_frame=None, stop_frame=None, cutout=[3., 3.]):
        '''This function returns the spike waveforms from the specified unit_id from t_start and t_stop
        in the form of a numpy array of spike waveforms.

        Parameters
        ----------
        unit_id: (int or list)
            The unit id or list of unit ids to extract waveforms from
        start_frame: (int)
            The starting frame to extract waveforms
        end_frame: (int)
            The ending frame to extract waveforms
        cutout: list of 2 floats
            Time in ms to cut out waveform before (cutout[0]) and after (cutout[1]) the peak.

        Returns
        -------
        waveforms: np.array
            A 3D array that contains all waveforms between start and end_frame (
            Dimensions are: (numm_spikes x num_channels x num_spike_frames)

        '''
        if isinstance(unit_id, int):
            unit_id = [unit_id]
        elif unit_id is None:
            unit_id = self.output_extractor.getUnitIds()
        elif not isinstance(unit_id, (list, np.array)):
            raise Exception("Unit_id is not a valid in valid")

        waveform_list = []
        for idx in unit_id:
            unit_ind = np.where(self.output_extractor.getUnitIds() == idx)[0]
            if len(unit_ind) == 0:
                raise Exception("Unit_id is not in valid")
            else:
                unit_ind = unit_ind[0]
                if self._waveforms[unit_ind] is None:
                    recordings = self.input_extractor.getRawTraces(start_frame, stop_frame)
                    fs = self.input_extractor.getSamplingFrequency()
                    times = np.arange(recordings.shape[1])
                    spike_times = self.output_extractor.getUnitSpikeTrain(unit_ind, start_frame, start_frame)

                    n_pad = [int(cutout[0] * fs / 1000), int(cutout[1] * fs / 1000)]

                    num_channels, num_frames = recordings.shape
                    num_spike_frames = np.sum(n_pad)

                    waveforms = np.zeros((len(spike_times), num_channels, num_spike_frames))
                    print('Number of waveforms: ', len(spike_times))

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


    def computeUnitTemplate(self, unit_id=None, start_frame=None, stop_frame=None):
        if isinstance(unit_id, int):
            unit_id = [unit_id]
        elif unit_id is None:
            unit_id = self.output_extractor.getUnitIds()
        elif not isinstance(unit_id, (list, np.array)):
            raise Exception("Unit_id is not a valid in valid")

        template_list = []
        for idx in unit_id:
            unit_ind = np.where(self.output_extractor.getUnitIds() == idx)[0]
            if len(unit_ind) == 0:
                raise Exception("Unit_id is not in valid")
            else:
                unit_ind = unit_ind[0]
            if self._templates[unit_ind] is None:
                if self._waveforms[unit_ind] is None:
                    self.getUnitWaveforms(unit_id, start_frame, stop_frame)
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


    def computeMaxChannel(self, unit_id=None):
        if isinstance(unit_id, int):
            unit_id = [unit_id]
        elif unit_id is None:
            unit_id = self.output_extractor.getUnitIds()
        elif not isinstance(unit_id, (list, np.array)):
            raise Exception("Unit_id is not a valid in valid")

        max_list = []
        for idx in unit_id:
            unit_ind = np.where(self.output_extractor.getUnitIds() == idx)[0]
            if len(unit_ind) == 0:
                raise Exception("Unit_id is not in valid")
            else:
                unit_ind = unit_ind[0]

            if self._maxchannels[unit_ind] is None:
                if self._templates[unit_ind] is None:
                    self.computeUnitTemplate(unit_id)
                max_chan = np.unravel_index(np.argmax(np.abs(self._templates[unit_ind])),
                                            self._templates[unit_ind].shape)[0]
                self._maxchannels[unit_ind] = max_chan
                max_list.append(max_chan)
            else:
                max_chan = self._maxchannels[unit_ind]
                max_list.append(max_chan)

        if len(max_list) == 1:
            return max_list[0]
        else:
            return max_list


    def computePCAscores(self, n_comp=10):
        from sklearn.decomposition import PCA

        # concatenate all waveforms
        all_waveforms = np.array([])
        nspikes = []
        for i_w, wf in enumerate(self._waveforms):
            if wf is None:
                self.getUnitWaveforms(self.output_extractor.getUnitIds()[i_w])
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

        return pca_scores, var


