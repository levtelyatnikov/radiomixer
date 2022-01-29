from torch import Tensor
from abc import abstractmethod

from radiomixer.io.signal import Signal
from radiomixer.transforms.transform import Transform

class Scaler(Transform):
    """Scaler is a base class for different types of concrete Scalers."""

    def process(self, signal: Signal) -> Signal:
        """Apply scaling to signal.

        :param signal: Signal object to normalise
        :return: Modified signal
        """
        signal.name = self._prepend_transform_name(signal.name)
        signal= self._scale(signal)
        
        return signal

    @abstractmethod
    def _scale(self, tensor: Tensor):
        """Concrete Scalers must implement this method. 
        
        In this method,the specific scaling strategy must be implemented.
        :param array: Array to scale
        :return: Scaled array
        """