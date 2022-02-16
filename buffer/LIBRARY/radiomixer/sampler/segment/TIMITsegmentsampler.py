import numpy as np
from radiomixer.sampler.segment.segment import SegmentGenerator
from radiomixer.transforms.transform import TransformType

class TIMITSegmentSilenceSampler(SegmentGenerator):
    
    def __init__(self, configs):
        super().__init__(TransformType.TIMITSEGMENTSILENCESAMPLER)

        self.SegmentSampler = TIMITSegmentSampler(configs)
        self.SilenceSampler = SilenceSampler(configs)
        

    def _sampler(self, signals: list):
        """"

        Take as an input 
        list of Signals
        """
        n_signals = len(signals)
        #signals_lengths = [signal.parameters['init_duration'] for signal in signals]
        phonems_list = [signal.parameters['phonems'] for signal in signals]

        silences_lengths = self.SilenceSampler.process(silences_number = n_signals-1)
        segments, silences_lengths, segments_phonem = self.SegmentSampler.process(silences_lengths = silences_lengths,
                                                                 phonems_list = phonems_list)

        new_signals = []
        for signal, segment, phonem in zip(signals, segments, segments_phonem):
            #print('segment = ', segment[1]-segment[0])
            signal.parameters['segment'] = segment
            signal.parameters['silences_lengths'] = silences_lengths
            signal.parameters['segment_phonem'] = phonem
            new_signals.append(signal)
            
        return new_signals





class TIMITSegmentSampler():
    def __init__(self, configs):
        self.configs = configs

        self.sampling_rate = self.configs['sampling_rate']
        self.audio_total_dur = int(self.configs['final_audio_clip_duration']*self.sampling_rate)

    def process(self, silences_lengths, phonems_list):
        self.silences_lengths = silences_lengths
        self.phonems_list = phonems_list
        segments, segments_phonem = self.segments_()

        return segments, self.silences_lengths, segments_phonem

    def segments_(self,):
        segments_number = len(self.silences_lengths) + 1
        self.segments_lengths, segments, segments_phonem = [], [], []
        silence_duration = np.sum(self.silences_lengths)
        spare_time = int(self.audio_total_dur - silence_duration)
        
        for idx in range(segments_number):
            
            phonems_audio = self.phonems_list[idx]
            check = False
            count_it = 0
            while check==False:
                count_it += 1
                segment, phonem  = phonems_audio[np.random.choice(len(phonems_audio), size=1)[0]]
                segment = list(segment)

                delta = segment[1] - segment[0]

                if spare_time > delta:
                    check = True
                
                if count_it>200 and spare_time<delta:
                    segment[1] = segment[0] + spare_time
                    delta = spare_time
                    check = True
                    print(f'CHeating delta={delta}')


            spare_time -= delta

            assert delta > 0, f"Segment duration is smaller or equal to 0, delta = {delta}" 
            self.segments_lengths.append(delta)
            segments.append(tuple(segment))
            segments_phonem.append(phonem)

        assert spare_time >= 0, f"Duration of segments longer then desired audio duration, {spare_time}"

        
        self.silence_update(spare_time)
        self.check_segments_(segments)

        return segments, segments_phonem

    def silence_update(self, spare_time):
        add_silence = (spare_time//len(self.silences_lengths)) + 1
        self.silences_lengths = [silence + add_silence for silence in self.silences_lengths]
        curr_audio_total_durr = np.sum(self.segments_lengths) + np.sum(self.silences_lengths)
        if curr_audio_total_durr > self.audio_total_dur:
            self.silences_lengths[-1] = int(self.silences_lengths[-1] - (curr_audio_total_durr - self.audio_total_dur))
       
        
    def check_segments_(self, segments):
        segments_lenths = np.array(list(map(lambda x: x[1] - x[0], segments)))
        curr_audio_total_duation = (np.sum(segments_lenths) + np.sum(self.silences_lengths))
        #assert np.all(segments_lenths > 0), f" Some of the segments smaller then minimum duration {segments_lenths}, minimum duration={self.segment_min}"
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
