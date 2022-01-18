import torch
import torchaudio
from radiomixer.transforms.transform import Transform, TransformType


class CustomFilter(Transform):
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
        durr = self._parameters['fade_in'] + self._parameters["stable_duration"] + self._parameters['fade_out']
        self.mask = torch.ones((1, durr))
        self._transition_in = torchaudio.transforms.Fade(fade_in_len = self._parameters['fade_in'],
                                                      fade_out_len = 0,
                                                      fade_shape = self._parameters['in_transition_type']
                                                      )

        self._transition_out = torchaudio.transforms.Fade(fade_in_len = 0,
                                                      fade_out_len = self._parameters['fade_out'],
                                                      fade_shape = self._parameters['out_transition_type']
                                                      )
    def _transition(self, audio):
        audio_dur = audio.shape[1]
        self.mask = self._transition_out(self._transition_in(self.mask))
        self.mask = torch.concat([ torch.zeros((1, self._parameters["silence_duration"])),  self.mask], dim=1)
        self.mask = torch.concat([self.mask, torch.zeros((1, audio_dur - self.mask.shape[1]))], dim=1)
        return audio * self.mask

    def process(self, signal):
        signal.name = self._prepend_transform_name(signal.name)
        self.parameters = signal.parameters['transition_parameters']
        
        signal.data = self._transition(signal.data)
        return signal
