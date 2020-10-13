import spikeextractors as se
import numpy as np
from tqdm import tqdm

from .utils import update_all_param_dicts_with_kwargs, select_max_channels_from_waveforms, \
    get_max_channels_per_waveforms, extract_snippet_from_traces, divide_recording_into_time_chunks, \
    get_unit_waveforms_for_chunk


def get_unit_waveforms2(
        recording,
        sorting,
        unit_ids=None,
        channel_ids=None,
        return_idxs=False,
        chunk_size=None,
        chunk_mb=50,
        **kwargs
):
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is is invalid")
    if isinstance(channel_ids, (int, np.integer)):
        channel_ids = [channel_ids]

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()

    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"
    assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Invalid channel_ids"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    grouping_property = params_dict['grouping_property']
    ms_before = params_dict['ms_before']
    ms_after = params_dict['ms_after']
    dtype = params_dict['dtype']
    max_spikes_per_unit = params_dict['max_spikes_per_unit']
    compute_property_from_recording = params_dict['compute_property_from_recording']
    memmap = params_dict['memmap']
    n_jobs = params_dict['n_jobs']
    joblib_backend = params_dict['joblib_backend']
    max_channels_per_waveforms = params_dict['max_channels_per_waveforms']
    seed = params_dict['seed']
    verbose = params_dict['verbose']
    save_property_or_features = params_dict['save_property_or_features']
    recompute_info = params_dict['recompute_info']

    waveform_list = []
    spike_index_list = []
    channel_index_list = []

    if 'waveforms' in sorting.get_shared_unit_spike_feature_names() and not recompute_info:
        for unit_id in unit_ids:
            waveforms = sorting.get_unit_spike_features(unit_id, 'waveforms')
            waveform_list.append(waveforms)
            if return_idxs:
                if len(waveforms) < len(sorting.get_unit_spike_train(unit_id)):
                    indexes = sorting.get_unit_spike_features(unit_id, 'waveforms_idxs')
                else:
                    indexes = np.arange(len(waveforms))
                if 'waveforms_channel_idxs' in sorting.get_shared_unit_property_names():
                    channel_idxs = sorting.get_unit_property(unit_id, 'waveforms_channel_idxs')
                else:
                    channel_idxs = np.arange(recording.get_num_channels())
                spike_index_list.append(indexes)
                channel_index_list.append(channel_idxs)
    else:

        if dtype is None:
            dtype = recording.get_dtype()

        # num_channels = recording.get_num_channels()
        num_frames = recording.get_num_frames()
        fs = recording.get_sampling_frequency()
        n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

        # set chunk size
        if chunk_size is not None:
            chunk_size = int(chunk_size)
        elif chunk_mb is not None:
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            max_size = int(chunk_mb * 1e6)  # set Mb per chunk
            chunk_size = max_size // (recording.get_num_channels() * n_bytes)

        # chunk_size = num_bytes_per_chunk / num_bytes_per_frame
        padding_size = 100 + n_pad[0] + n_pad[1]  # a bit excess padding
        chunks = divide_recording_into_time_chunks(
            num_frames=num_frames,
            chunk_size=chunk_size,
            padding_size=padding_size
        )
        n_chunk = len(chunks)

        # pre-map memmap files
        n_channels = get_max_channels_per_waveforms(recording, grouping_property, channel_ids,
                                                    max_channels_per_waveforms)

        if memmap:
            all_unit_waveforms = []
            for unit_id in unit_ids:
                fname = 'waveforms_' + str(unit_id) + '.raw'
                len_wf = len(sorting.get_unit_spike_train(unit_id))
                if max_spikes_per_unit is not None:
                    if len_wf > max_spikes_per_unit:
                        len_wf = max_spikes_per_unit
                shape = (len_wf, n_channels, sum(n_pad))
                arr = sorting.allocate_array(shape=shape, dtype=dtype, name=fname, memmap=memmap)
                all_unit_waveforms.append(arr)
        else:
            all_unit_waveforms = [[] for ii in range(len(unit_ids))]

        if verbose:
            chunk_iter = tqdm(range(n_chunk), ascii=True, desc="Extracting waveforms in chunks")
        else:
            chunk_iter = range(n_chunk)

        if max_spikes_per_unit is not None:
            for i, unit in enumerate(unit_ids):
                num_spikes = len(sorting.get_unit_spike_train(unit))
                if num_spikes > max_spikes_per_unit:
                    spike_index_list.append(np.sort(np.random.permutation(num_spikes)[:max_spikes_per_unit]))
                else:
                    spike_index_list.append(np.arange(num_spikes))

        wf_chunk_idxs = np.zeros(len(unit_ids), dtype='int')
        cumulative_spikes = np.zeros(len(unit_ids), dtype='int')

        for ii in chunk_iter:
            chunk = chunks[ii]
            # chunk: {istart, iend, istart_with_padding, iend_with_padding} # include padding
            recording_chunk = se.SubRecordingExtractor(
                parent_recording=recording,
                start_frame=chunk['istart_with_padding'],
                end_frame=chunk['iend_with_padding']
            )
            # note that the efficiency of this operation may need improvement
            # (really depends on sorting extractor implementation)
            sorting_chunk = se.SubSortingExtractor(
                parent_sorting=sorting,
                start_frame=chunk['istart'],
                end_frame=chunk['iend']
            )

            # num_events_in_chunk x num_channels_in_nbhd[unit_id] x len_of_one_snippet
            unit_waveforms = get_unit_waveforms_for_chunk(
                recording=recording_chunk,
                sorting=sorting_chunk,
                frame_offset=chunk['istart'] - chunk['istart_with_padding'],
                unit_ids=unit_ids,
                snippet_len=n_pad,
                channel_ids_by_unit=None,
                waveform_indexes=spike_index_list,
                spikes_so_far=cumulative_spikes
            )

            # TODO fix max channels. If chunk is too short, zero waveforms could be found...
            # Either 1) compute full waveform and filter at the end
            # Or: 2) Keep waveforms in memory until a min number (100) is reached, then dump to memmap
            for i_unit, unit in enumerate(unit_ids):
                cumulative_spikes[i_unit] += len(sorting_chunk.get_unit_spike_train(unit))
                wf = unit_waveforms[i_unit]
                wf = wf.astype(dtype)

                # compute waveform channels from first chunk
                if ii == 0:
                    if max_channels_per_waveforms < len(channel_ids):
                        max_channel_idxs = select_max_channels_from_waveforms(wf, recording, max_channels_per_waveforms)
                    else:
                        max_channel_idxs = np.arange(len(channel_ids))
                    channel_index_list.append(max_channel_idxs)

                wf = wf[:, channel_index_list[i_unit]]
                if memmap:
                    all_unit_waveforms[i_unit][wf_chunk_idxs[i_unit]:wf_chunk_idxs[i_unit] + len(wf)] = wf
                    wf_chunk_idxs[i_unit] += len(wf)
                else:
                    all_unit_waveforms[i_unit].append(wf)

        if memmap:
            waveform_list = all_unit_waveforms
        else:
            # concatenate the results over the chunks
            waveform_list = [
                # tot_num_events_for_unit x num_channels_in_nbhd[unit_id] x len_of_one_snippet
                np.concatenate(all_unit_waveforms[i_unit], axis=0)
                for i_unit in range(len(unit_ids))
            ]

        # return correct max channels

        if save_property_or_features:
            for i, unit_id in enumerate(unit_ids):
                sorting.set_unit_spike_features(unit_id, 'waveforms', waveform_list[i], indexes=spike_index_list[i])
                if len(channel_index_list[i]) < recording.get_num_channels():
                    sorting.set_unit_property(unit_id, 'waveforms_channel_idxs', channel_index_list[i])

    return waveform_list
