from abc import ABC, abstractmethod
import spikeextractors as se
import numpy as np
import threading
import time


class filterChunkThread(threading.Thread):
    def __init__(self, chunk_id, recording, ich1, ich2, start_frame, end_frame, channel_ids):
        threading.Thread.__init__(self)
        self.chunk_id = chunk_id
        self.recording = recording
        self.ich1 = ich1
        self.ich2 = ich2
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.channel_ids = channel_ids
        self.filtered_chunk = None

    def run(self):
        filtered_chunk0 = self.recording._get_filtered_chunk(self.chunk_id)
        if self.chunk_id == self.ich1:
            start0 = self.start_frame - self.chunk_id * self.recording._chunk_size
        else:
            start0 = 0
        if self.chunk_id == self.ich2:
            end0 = self.end_frame - self.chunk_id * self.recording._chunk_size
        else:
            end0 = self.recording._chunk_size
        chan_idx = [self.recording.getChannelIds().index(chan) for chan in self.channel_ids]
        self.filtered_chunk = filtered_chunk0[chan_idx, start0:end0]


class FilterRecording(se.RecordingExtractor):
    def __init__(self, *, recording, chunk_size=10000):
        se.RecordingExtractor.__init__(self)
        self._recording = recording
        self._chunk_size = chunk_size
        self._filtered_chunk_cache = FilteredChunkCache()
        self.copyChannelProperties(recording)

    def getChannelIds(self):
        return self._recording.getChannelIds()

    def getNumFrames(self):
        return self._recording.getNumFrames()

    def getSamplingFrequency(self):
        return self._recording.getSamplingFrequency()

    def getTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        ich1 = int(start_frame / self._chunk_size)
        ich2 = int((end_frame - 1) / self._chunk_size)
        filtered_chunk_list = []

        filt_threads = []
        for ich in range(ich1, ich2 + 1):
            filt_threads.append(filterChunkThread(ich, self, ich1, ich2,start_frame, end_frame, channel_ids))
        for t in filt_threads:
            t.start()
        for t in filt_threads:
            t.join()

        sorted_threads = np.array(filt_threads)[np.argsort([t.chunk_id for t in filt_threads])]
        for t in sorted_threads:
            filtered_chunk_list.append(t.filtered_chunk)

        return np.concatenate(filtered_chunk_list, axis=1)

    @abstractmethod
    def filterChunk(self, *, start_frame, end_frame):
        raise NotImplementedError('filterChunk not implemented')

    def _get_filtered_chunk(self, ind):
        code = str(ind)
        chunk0 = self._filtered_chunk_cache.get(code)
        if chunk0 is not None:
            return chunk0
        else:
            start0 = ind * self._chunk_size
            end0 = (ind + 1) * self._chunk_size
            chunk1 = self.filterChunk(start_frame=start0, end_frame=end0)
            self._filtered_chunk_cache.add(code, chunk1)
            return chunk1


class FilteredChunkCache():
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
