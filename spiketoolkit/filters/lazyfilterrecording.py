from abc import ABC, abstractmethod
import spikeinterface as si
import numpy as np

class FilterRecording(si.RecordingExtractor):
    def __init__(self, *, recording,chunk_size=10000):
        si.RecordingExtractor.__init__(self)
        self._recording=recording
        self._chunk_size=chunk_size
        self._filtered_chunks=dict()
        self.getNumChannels=recording.getNumChannels
        self.copyChannelProperties(recording)
        
    def getNumChannels(self):
        return self._recording.getNumChannels()
    
    def getNumFrames(self):
        return self._recording.getNumFrames()
    
    def getSamplingFrequency(self):
        return self._recording.getSamplingFrequency()
        
    def getTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self.getNumFrames()
        if channel_ids is None:
            channel_ids=list(range(self.getNumChannels()))
        ich1=int(start_frame/self._chunk_size)
        ich2=int((end_frame-1)/self._chunk_size)
        filtered_chunk_list=[]
        for ich in range(ich1,ich2+1):
            filtered_chunk0=self._get_filtered_chunk(ich)
            if ich==ich1:
                start0=start_frame-ich*self._chunk_size
            else:
                start0=0
            if ich==ich2:
                end0=end_frame-ich*self._chunk_size
            else:
                end0=self._chunk_size
            filtered_chunk_list.append(filtered_chunk0[channel_ids,start0:end0])
        return np.concatenate(filtered_chunk_list,axis=1)
    
    @abstractmethod
    def filterChunk(self,*,start_frame,end_frame):
        raise NotImplementedError('filterChunk not implemented')
    
    def _get_filtered_chunk(self, ind):
        code=str(ind)
        if code not in self._filtered_chunks:
            start0=ind*self._chunk_size
            end0=(ind+1)*self._chunk_size
            self._filtered_chunks[code]=self.filterChunk(start_frame=start0,end_frame=end0)
        return self._filtered_chunks[code]
    
