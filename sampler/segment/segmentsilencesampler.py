import numpy as np
from radiomixer.sampler.segment.segment import SegmentGenerator
from radiomixer.transforms.transform import TransformType

class SegmentSilenceSampler(SegmentGenerator):
    
    def __init__(self, configs):
        super().__init__(TransformType.SEGMENTSILENCESAMPLER)

        self.SegmentSampler = SegmentSampler(configs)
        self.SilenceSampler = SilenceSampler(configs)
        

    def _sampler(self, signals: list):
        """"

        Take as an input 
        list of Signals
        """
        n_signals = len(signals)
        signals_lengths = [signal.parameters['init_duration'] for signal in signals]

        silences_lengths = self.SilenceSampler.process(silences_number = n_signals-1)
        segments, silences_lengths = self.SegmentSampler.process(silences_lengths=silences_lengths,
                                                              audio_lengths=signals_lengths)

        new_signals = []
        for signal, segment in zip(signals, segments):
            signal.parameters['segment'] = segment
            signal.parameters['silences_lengths'] = silences_lengths
            new_signals.append(signal)
            
        return new_signals





class SegmentSampler():
    def __init__(self, configs):
        self.configs = configs

        self.sr = self.configs['sampling_rate']
        self.audio_total_dur = int(self.configs['final_audio_clip_duration']*self.sr)
        self.segment_min = int(self.configs['segment_min_duration']*self.sr)
        self.segment_max = int(self.configs['segment_max_duration']*self.sr)

    def process(self, silences_lengths, audio_lengths):
        self.silences_lengths = silences_lengths
        self.audio_lengths = audio_lengths

        self.get_segments_lengths()
        segments = self.segments_()

        return segments, self.silences_lengths

    def get_segments_lengths(self,):
        segments_number = len(self.audio_lengths)
        #print('segments_number', segments_number)
        self.segments_lengths = []
        silence_duration = np.sum(self.silences_lengths)
        #print('silence_duration', silence_duration)
        spare_time = int(self.audio_total_dur - (segments_number*self.segment_min + silence_duration))
        #print('(segments_number*self.segment_min + silence_duration)', (segments_number*self.segment_min + silence_duration))

        for idx in range(segments_number):
            
            # chose right bound to consider the case
            # when audio duration is equal to minimum segment length
            
            #high = min(spare_time+1, self.audio_lengths[idx] - self.segment_min + 1)
            high = min(spare_time+1, min(self.segment_max, self.audio_lengths[idx] - self.segment_min + 1))
            # print("spare_time+1", spare_time+1)
            # print('self.segment_max', self.segment_max)
            # print("self.audio_lengths[idx]", self.audio_lengths[idx])
            # print('self.audio_lengths[idx] - self.segment_min + 1', self.audio_lengths[idx] - self.segment_min + 1)
            # print('high', high)
            

            
            s_delta = np.random.randint(low=0, high=high)

            self.segments_lengths.append(self.segment_min + s_delta)
            spare_time -= s_delta

        
        self.silence_update(spare_time)
        
    

    def segments_(self,):
        segments = []
        for audio_length, seg_length in zip(self.audio_lengths, self.segments_lengths):
            
            start = np.random.randint(low=0, high=audio_length - seg_length+1)
            end = int(start + seg_length)
            segments.append( (start, end) )

        self.check_segments_(segments)
        return segments

    def silence_update(self, spare_time):
        add_silence = (spare_time//len(self.silences_lengths)) + 1
        self.silences_lengths = [silence + add_silence for silence in self.silences_lengths]
        curr_audio_total_durr = np.sum(self.segments_lengths) + np.sum(self.silences_lengths)
        if curr_audio_total_durr > self.audio_total_dur:
            self.silences_lengths[-1] = int(self.silences_lengths[-1] - (curr_audio_total_durr - self.audio_total_dur))
       
        
    def check_segments_(self, segments):
        segments_lenths = np.array(list(map(lambda x: x[1] - x[0], segments)))
        curr_audio_total_duation = (np.sum(segments_lenths) + np.sum(self.silences_lengths))
        assert np.all(segments_lenths >= self.segment_min), f" Some of the segments smaller then minimum duration {segments_lenths}, minimum duration={self.segment_min}"
        assert curr_audio_total_duation == self.audio_total_dur, f"Total duration is wrong: {curr_audio_total_duation} != {self.audio_total_dur}"


class SilenceSampler():
    def __init__(self, configs):
        self.configs = configs

        self.sr = self.configs['sampling_rate']
        # Parameters to generate silence
        self.s_total = int(self.configs['max_total_silence_dur']*self.sr)
        self.s_min = int(self.configs['min_segment_silence_dur']*self.sr)


    def process(self, silences_number, random_max_silence=True):
        silence_segments = []
        min_silence =  silences_number * self.s_min
      
        if random_max_silence:
            
            s_total = np.random.randint(low=min_silence, high=self.s_total+1)
        else:
            s_total = self.s_total

        spare_time = s_total - min_silence 
        
        for _ in range(silences_number):
            # sample duration of the silence
            # total segment silence duaration = s_min + s_delta
            s_delta = np.random.randint(low=0, high=spare_time+1)

            # update spare time
            spare_time = spare_time - s_delta
           
            silence_segments.append(int(self.s_min + s_delta))
            
        # shuffle silences 
        np.random.shuffle(silence_segments)
        self.silence_property_check_(silence_segments)
        
        return silence_segments

    def silence_property_check_(self, arr):
        arr = np.array(arr)
        assert np.all(arr >= self.s_min), f"Minimum silence segment error {arr}"
