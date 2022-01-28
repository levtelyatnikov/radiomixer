
"""
This module contains TransformFactory, a class that enables to
instantiate transform objects.
"""

import logging

from radiomixer.transforms.transform import Transform
from radiomixer.transforms.mixer.mixers import CustomMixer, CustomFilter
from radiomixer.transforms.scaler.scalers import LibrosaNorm, MinMaxScaler
from radiomixer.transforms.feature.MelSpectrogram import MelSpectrogram
from radiomixer.transforms.filters.torchfilters import TorchFilterIn, TorchFilterOut
from radiomixer.transforms.concatenator.concatenators import SequentialConcatenator, SummationConcatenator


from radiomixer.sampler.segment.segmet_samplers_wrappers import EqualSegmentSamplerWrapper
from radiomixer.sampler.transition.transitions import TransitionOverlapedSegmentsParametersSampler

logger = logging.getLogger(__name__)

class TransformFactory:
    """Factory that instantiates Transform objects.
    
    A class that enables to instantiate transform objects.
    """

    def __init__(self):
        self.transform_types = {
            "TorchFilterIn": TorchFilterIn,
            "TorchFilterOut": TorchFilterOut,
            "MinMaxScaler": MinMaxScaler,
            "LibrosaNormalize": LibrosaNorm,
            "SequentialConcatenator": SequentialConcatenator,
            
            "MelSpectrogram": MelSpectrogram,
            
            "TransitionOverlapedSegmentsParametersSampler": TransitionOverlapedSegmentsParametersSampler,
            "EqualSegmentSampler": EqualSegmentSamplerWrapper,

            "CustomMixer": CustomMixer,
            "CustomFilter":CustomFilter,
            "SummationConcatenator": SummationConcatenator
        }
        #"ExtractSegment": ExtractSegment,
        #"ExtractAndFade": ExtractAndFade,
        logger.info("Initialised TransformFactory object")

    def create(self, transform_type: str, **kwargs) -> Transform:
        """Instantiate and return concrete transform.

        :param transform_type: Type of transform object to instantiate
        :return: Instance of concrete transform
        """
        self._raise_type_error_if_transform_isnt_avaialbe(transform_type)
        transform = self.transform_types.get(transform_type)
        if kwargs != {}:
            
            transform_module = transform(kwargs)
        else:
            transform_module = transform()

        return transform_module

    def _raise_type_error_if_transform_isnt_avaialbe(self, transform_type):
        if transform_type not in self.transform_types:
            raise TypeError("It's not possible to instantiate a "
                            f"'{transform_type}' beause this transform "
                            "doesn't exist.")