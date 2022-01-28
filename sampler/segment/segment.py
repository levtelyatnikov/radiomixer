from abc import abstractmethod
from radiomixer.transforms.transform import TransformSeq2Seq, TransformType

class SegmentGenerator(TransformSeq2Seq):
    """
    
    SegmentGenerator takes list of Signals.
    Generate segment for each Signal.
    The generation is conditioned on transition properties.
    Hence it is needed to generate initial point of segment
    and segment duration.
    """
    
    def process(self, signals: list):
        """

        Takes list of signals as an input, generates
        signals. Add corresponding segments into 
        corresponding signal object.
        """

        signals = self._sampler(signals)
        for signal in signals:  
            signal.name = self._prepend_transform_name(signal.name)
            
        return signals
    
    @abstractmethod
    def _sampler(self):
        pass

        

