from abc import abstractmethod
from radiomixer.io.signal import Signal, SignalFeature
from radiomixer.transforms.transform import TransformSeq2Signal



class Concatenator(TransformSeq2Signal):
    """Concatenator is responsible for signals concatenation"""

    def process(self, signals: list) -> SignalFeature:
        """Apply concatenation to signals.

        :param signal: Signals list 
        :return: signal
        """
        
        signal = self._concatenate(signals)
        signal.name = self._prepend_transform_name(signals[-1].name)
        return signal

    @abstractmethod
    def _concatenate(self, signals: list):
        pass
