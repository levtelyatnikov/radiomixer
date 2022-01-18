import numpy as np
from torch import Tensor
from librosa.util import normalize

from radiomixer.io.signal import SignalFeature
from radiomixer.transforms.scaler.scaler import Scaler
from radiomixer.transforms.transform import TransformType

class LibrosaNorm(Scaler):
    def __init__(self):
        super().__init__(TransformType.LIBROSANORMALIZE)

    def _scale(self, signal):
        """Applicable to Signal and SignalFeature object"""

        signal.data = Tensor(normalize(signal.data.numpy(), axis=1))
        return signal

# ----------------------------MinMaxScaler------------------------------------

class MinMaxScaler(Scaler):
    """MinMaxScaler performs min max scaling on a audio.
    
    It's a concrete Scaler. If the signal is 2-dimensional
    (e.g., spectrogram), the mean / std deviation are calculated
    across all rows.
    Attributes:
        - min_val: Lowest scaling range
        - max_val: Highest scaling range
    If the signal is 2-dimensional (e.g., spectrogram), the mean / std
    deviation are calculated gloabally across all rows.
    """

    def __init__(self, min: float = 0., max: float = 1.):
        super().__init__(TransformType.MINMAXSCALER)
        self.min_val = min
        self.max_val = max

    def _scale(self, signal: SignalFeature) -> SignalFeature:
        """
        
        This class is applicable only to SignalFeature
        however it can be easily modified.
        """
        
        min_val, max_val = signal.data_features.min(), signal.data_features.max()
        signal.parameters_own['min'], signal.parameters_own['max'] = np.round(min_val.item(),4), np.round(max_val.item(),4)

        signal.data_features = (signal.data_features - min_val) / (max_val - min_val)
        signal.data_features = signal.data_features * (self.max_val - self.min_val) + self.min_val
        return signal