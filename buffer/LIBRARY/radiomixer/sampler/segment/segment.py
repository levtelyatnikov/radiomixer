from abc import ABC, abstractmethod
from radiomixer.transforms.transform import TransformSeq, TransformType

class SegmentGenerator(TransformSeq):

    
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

        

