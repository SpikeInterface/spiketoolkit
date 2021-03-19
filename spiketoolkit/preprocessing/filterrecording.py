from abc import abstractmethod
import numpy as np
from .transform import TransformRecording
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args


class FilterRecording(BasePreprocessorRecordingExtractor):
    def __init__(self, recording, chunk_size=10000, cache_chunks=False, dtype=None):
        self._chunk_size = chunk_size
        self._cache_chunks = cache_chunks
        if cache_chunks:
            self._filtered_cache_chunks = FilteredChunkCache()
        else:
            self._filtered_cache_chunks = None
        self._traces = None
        if dtype is None:
            dtype = str(recording.get_dtype())
        if 'uint' in dtype:
            if 'numpy' in dtype:
                dtype = str(dtype).replace("<class '", "").replace("'>", "")
                # drop 'numpy'
                dtype = dtype.split('.')[1]
            dtype_signed = dtype[1:]
            exp_idx = dtype.find('int') + 3
            exp = int(dtype[exp_idx:])
            offset = - 2**(exp - 1)
            recording_base = TransformRecording(recording, offset=offset, dtype=dtype_signed)
            print(f"dtype converted from {dtype} to {dtype_signed} before filtering")
            self._dtype = dtype_signed
        else:
            self._dtype = dtype
            recording_base = recording
        BasePreprocessorRecordingExtractor.__init__(self, recording_base)

    # avoid filtering one sample
    def get_dtype(self, return_scaled=True):
        return self._dtype

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        if self._chunk_size is not None:
            ich1 = int(start_frame / self._chunk_size)
            ich2 = int((end_frame - 1) / self._chunk_size)
            dt = self.get_dtype()
            filtered_chunk = np.zeros((len(channel_ids), int(end_frame-start_frame)), dtype=dt)
            pos = 0
            for ich in range(ich1, ich2 + 1):
                filtered_chunk0 = self._get_filtered_chunk(ich, channel_ids, return_scaled)
                if ich == ich1:
                    start0 = start_frame - ich * self._chunk_size
                else:
                    start0 = 0
                if ich == ich2:
                    end0 = end_frame - ich * self._chunk_size
                else:
                    end0 = self._chunk_size
                filtered_chunk[:, pos:pos+end0-start0] = filtered_chunk0[:, start0:end0]
                pos += (end0-start0)
        else:
            filtered_chunk = self.filter_chunk(start_frame=start_frame, end_frame=end_frame, channel_ids=channel_ids,
                                               return_scaled=return_scaled)
        return filtered_chunk.astype(self._dtype)

    @abstractmethod
    def filter_chunk(self, *, start_frame, end_frame, channel_ids, return_scaled):
        raise NotImplementedError('filter_chunk not implemented')

    def _read_chunk(self, i1, i2, channel_ids, return_scaled=True):
        num_frames = self._recording.get_num_frames()
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > num_frames:
            i2b = num_frames
        else:
            i2b = i2
        chunk = np.zeros((len(channel_ids), i2 - i1))
        chunk[:, i1b - i1:i2b - i1] = self._recording.get_traces(start_frame=i1b, end_frame=i2b,
                                                                 channel_ids=channel_ids, return_scaled=return_scaled)

        return chunk

    def _get_filtered_chunk(self, ind, channel_ids, return_scaled):
        if self._cache_chunks:
            code = str(ind)
            chunk0 = self._filtered_cache_chunks.get(code)
        else:
            chunk0 = None

        if chunk0 is not None:
            if chunk0.shape[0] == len(channel_ids):
                return chunk0
            else:
                channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
                return chunk0[channel_idxs]

        start0 = ind * self._chunk_size
        end0 = (ind + 1) * self._chunk_size

        if self._cache_chunks:
            # filter all channels if cache_chunks is used
            chunk1 = self.filter_chunk(start_frame=start0, end_frame=end0, channel_ids=self.get_channel_ids())
            self._filtered_cache_chunks.add(code, chunk1)
            channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
            chunk1 = chunk1[channel_idxs]
        else:
            # otherwise, only filter requested channels
            chunk1 = self.filter_chunk(start_frame=start0, end_frame=end0, channel_ids=channel_ids,
                                       return_scaled=return_scaled)

        return chunk1
            

class FilteredChunkCache:
    def __init__(self):
        self._chunks_by_code = dict()
        self._codes = []
        self._total_size = 0
        self._max_size = 1024 * 1024 * 100

    def add(self, code, chunk):
        self._chunks_by_code[code] = chunk
        self._codes.append(code)
        self._total_size = self._total_size + chunk.size
        if self._total_size > self._max_size:
            ii = 0
            while (ii < len(self._codes)) and (self._total_size > self._max_size / 2):
                self._total_size = self._total_size - self._chunks_by_code[self._codes[ii]].size
                del self._chunks_by_code[self._codes[ii]]
                ii = ii + 1
            self._codes = self._codes[ii:]

    def get(self, code):
        if code in self._chunks_by_code:
            return self._chunks_by_code[code]
        else:
            return None
