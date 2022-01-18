
import torchaudio
from radiomixer.io.signal import Signal
from radiomixer.transforms.transform import Transform, TransformType


class TorchFilterIn(Transform):
    """

    This class aims to modify the audio
    with help of torchaudio.transforms.Fade
    method. See the documentation for the set
    of available _parameters dict
    """

    def __init__(self):
        super().__init__(TransformType.TORCHFILTERIN)
        self._parameters = None
       
    @property
    def parameters(self):
        return self._parameters
       
    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        self._transition = torchaudio.transforms.Fade(fade_in_len = self._parameters['fade_in'],
                                                      fade_out_len = 0,
                                                      fade_shape = self._parameters['in_transition_type']
                                                      )
    def process(self, signal):
        signal.name = self._prepend_transform_name(signal.name)
        self.parameters = signal.parameters['transition_parameters']
        signal.data = self._transition(signal.data)
        return signal

    
class TorchFilterOut(Transform):
    """

    This class aims to modify the audio
    with help of torchaudio.transforms.Fade
    method. See the documentation for the set
    of available _parameters dict
    """

    def __init__(self):
        super().__init__(TransformType.TORCHFILTEROUT)
        self._parameters = None
       
    @property
    def params(self):
        return self._parameters
       
    @params.setter
    def params(self, parameters):
        self._parameters = parameters
        self._transition = torchaudio.transforms.Fade(fade_in_len = 0,
                                                      fade_out_len = self._parameters['fade_out'],
                                                      fade_shape = self._parameters['out_transition_type']
                                                      )
    def process(self, signal):
        self.params = signal.parameters['transition_parameters']
        signal.name = self._prepend_transform_name(signal.name)
        signal.data = self._transition(signal.data)
        return signal

