
import numpy as np
from radiomixer.utils.utils import sec2rate

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
        self.totalSignalDur = sec2rate(self.configs['final_audio_clip_duration'], self.sample_rate) 
        self.minSegmentDur = sec2rate(self.configs['segment_min_duration'], self.sample_rate)

    def process(self, signals: list):
        segments = []
        for signal in signals:
            dur_signal = signal.data.shape[1]
            dur_segment = self.sample_segment_dur(dur_signal=dur_signal)
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

    def sample_segment_dur(self, dur_signal):
        # Segment can't be longer then total audio duration hence
        hign = min(dur_signal, self.totalSignalDur)

        # Segment have to be longer or equal to minimum segment duration 
        dur_segment = hign
        assert dur_signal >= dur_segment, f"Sampled segment duation greater then current audio duration: dur_signal = {dur_signal}, segment_dur = {dur_segment}"
        return  dur_segment




