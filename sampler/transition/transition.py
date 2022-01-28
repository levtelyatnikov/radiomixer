
from abc import abstractmethod
from radiomixer.transforms.transform import TransformSeq2Seq, TransformType

class TransitionSampler(TransformSeq2Seq):
    """Abstract class that provides an interface to generate transitions"""
    
    def process(self, signals: list):

        for signal in signals:  
            
            signal.name = self._prepend_transform_name(signal.name)
            signal = self._sampler(signal)
        
        return signals

    @abstractmethod
    def _sampler(self):
        pass
