import numpy as np
from radiomixer.sampler.segment.segment import SegmentGenerator
from radiomixer.transforms.transform import TransformType
from radiomixer.utils.utils import sec2rate

class OverlapedEqualSegmentSampler(SegmentGenerator):
    
    def __init__(self, configs):
        super().__init__(TransformType.OVERLAPEDEQUALSEGMENTSAMPLER)

        self.SegmentSampler = EqualSegmentSampler(configs)

    def _sampler(self, signals: list):
        """"

        Take as an input 
        list of Signals
        """

        segments = self.SegmentSampler.process(signals)

        new_signals = []
        for signal, segment in zip(signals, segments):
            signal.parameters['segment'] = segment
            signal.parameters['segment_length'] = segment[1] - segment[0]
            new_signals.append(signal)
            
        return new_signals





class EqualSegmentSampler():
    """
    Core idea of this sampler is to sample segments
    which are equal to final_audio_clip_duration.
    In case when input audio is smaller then 
    final_audio_clip_duration, then full audio is taken.
    However such cases have to be omitted with augmenting input audio 
    during loading.
    """

    def __init__(self, configs):
        self.configs = configs

        self.sample_rate = self.configs['sampling_rate']
        # self.maxOverlapSegmets = self.configs['max_overlap_segments'] # percentage: [0, 1]
        # self.minOverlapSegment = self.configs['min_overlap_segments']
        self.totalSignalDur = sec2rate(self.configs['final_audio_clip_duration'], self.sample_rate) 
        self.minSegmentDur = sec2rate(self.configs['segment_min_duration'], self.sample_rate)
        
        

    def process(self, signals: list):
        segments = []
        for signal in signals:
            dur_signal = signal.data.shape[1]
            dur_segment = self.sample_segment_dur(dur_signal = dur_signal)
            segment = self.sample_segment(signal, dur_segment)
            segments.append(segment)

        return segments
            
    def sample_segment(self, signal, dur_segment):
        audio = signal.data
        audio_dur = audio.shape[1]
        high = audio_dur - dur_segment
        start = np.random.randint(low = 0, high = high+1)
        end = start + dur_segment
        assert (start <= audio_dur) and (end <= audio_dur)

        return (start, end)


    # def sample_overlap(self):
    #     return np.random.uniform(low = self.minOverlapSegment, high = self.maxOverlapSegmets)

    def sample_segment_dur(self, dur_signal):
        # Segment can't be longer then total audio duration hence
        # when idx == 0 then self.totalSignalDur
        
        hign = min(dur_signal, self.totalSignalDur)
        
        # Segment have to be longer or equal to minimum segment duration 
        #low = self.minSegmentDur
        dur_segment = hign
        #dur_segment = np.random.randint(low = low, high = hign)
        assert dur_signal >= dur_segment, f"Sampled segment duation greater then current audio duration: dur_signal = {dur_signal}, segment_dur = {dur_segment}"
        
        return  dur_segment




