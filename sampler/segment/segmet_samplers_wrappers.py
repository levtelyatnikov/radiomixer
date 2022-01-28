from radiomixer.sampler.segment.segment_samplers import EqualSegmentSampler
from radiomixer.sampler.segment.segment import SegmentGenerator
from radiomixer.transforms.transform import TransformType

class EqualSegmentSamplerWrapper(SegmentGenerator):
    
    def __init__(self, configs):
        super().__init__(TransformType.EQUALSEGMENTSAMPLER)

        self.SegmentSampler = EqualSegmentSampler(configs)

    def _sampler(self, signals: list) -> list:
        """
        
        Apply Segment sampler procedure to all signals in the list.
        Additionally update information in the signal.parameters.
        The information added into signal.parameters depends on 
        the needs. 
        """
        segments = self.SegmentSampler.process(signals)

        new_signals = []
        for signal, segment in zip(signals, segments):
            signal.parameters['segment'] = segment
            signal.parameters['segment_length'] = segment[1] - segment[0]
            new_signals.append(signal) 
        
        return new_signals