from radiomixer.io.signal import Signal
from radiomixer.transforms.mixer.mixer import Mixer
from radiomixer.transforms.transform import TransformType

from radiomixer.transforms.mixer.extractor import ExtractSegment
from radiomixer.transforms.filters.torchfilters import TorchFilterIn, TorchFilterOut 
from radiomixer.transforms.filters.customefilters import CustomFilter
from radiomixer.transforms.scaler.scalers import LibrosaNorm

class CustomMixer(Mixer):
    """CustomMixer class to Extract, Fade, Normalize segment


    CustomMixer class is responsible three things: 
      - Extract segment from audio
      - Apply generated transitions with help of CustomFilter
      - Normalize audio after transitions
    """
    def __init__(self):
        super().__init__(TransformType.CustomMixer)
        self.extractor = ExtractSegment()
        self.fade = CustomFilter()
        self.normalization = LibrosaNorm()
    
    def _pocess(self, signal: Signal) -> Signal:
        signal = self.extractor.process(signal)
        signal = self.fade.process(signal)
        signal = self.normalization.process(signal)
        return signal

    def process(self, signals):
        return  [self._pocess(signal) for signal in signals]

# ----------------------------TorchMixer--------------------------------------

class TorchMixer(Mixer):
    """CustomMixer class to Extract, Fade, Normalize segment


    CustomMixer class is responsible three things: 
      - Extract segment from audio
      - Apply generated transitions with help of TorchFilters
      - Normalize audio after transitions
    """
    def __init__(self):
        super().__init__(name=TransformType.TorchMixer)

        self.extractor = ExtractSegment()
        self.fade_in = TorchFilterIn()
        self.fade_out = TorchFilterOut()
        self.normalization = LibrosaNorm()
    
    def _pocess(self, signal: Signal) -> Signal:
        signal = self.extractor.process(signal)
        signal = self.fade_in.process(signal)
        signal = self.fade_out.process(signal)
        signal = self.normalization.process(signal)
        return signal

    def process(self, signals):
        return  [self._pocess(signal) for signal in signals]


