from abc import ABC, abstractmethod
from enum import Enum
from radiomixer.io.signal import Signal

import logging

logger = logging.getLogger(__name__)

class TransformType(Enum):
    """Enumeration class with all available transforms."""
    
    TORCHFILTERIN = 'TorchFilterIn'
    TORCHFILTEROUT = 'TorchFilterOut'
    LIBROSANORMALIZE = 'LibrosaNormalize'

    MELSPECTROGRAM = 'MelSpectrogram'
    MINMAXSCALER = 'MinMaxScaler'

    EXTRACTSEGMENT = 'ExtractSegment'
    TRANSITIONSAMPLER = 'TransitionSampler'
    EXTRACTANDFADE = 'ExtractAndFade'
    SEGMENTSILENCESAMPLER = 'SegmentSilenceSampler' 
    TIMITSEGMENTSILENCESAMPLER = 'TIMITSegmentSilenceSampler'
    OVERLAPEDEQUALSEGMENTSAMPLER = 'OverlapedEqualSegmentSampler'

    TRANSITIONSEGMENTSILENCESPARAMETERSSAMPLER = 'TransitionSegmentSilenceParametersSampler'
    TRANSITIONSOVERLAPEDSEGMENTSPARAMETERSSAMPLER = 'TransitionOverlapedSegmentsParametersSampler'

    SEQUENTIALCONCATENATOR = 'FequentialConcatenator'
    TorchMixer = 'TorchMixer'
    CustomMixer = 'CustomMixer'
    SummationConcatenator = 'SummationConcatenator'

class Transform(ABC):
    """Transform is a common interface for all transforms objects. Such
    objects manipulate a signal (e.g., applying log scaling, extracting
    MFCCs).
    Attrs:
        - name: The name of the transforms
    """

    def __init__(self, name: TransformType):
        self.name = name
        logger.info("Instantiated %s transform", self.name)
        
    @abstractmethod
    def process(self, signal: Signal) -> Signal:
        """This method is responsible to apply a transforms to the incoming
        signal.
        :param signal: Signal object to be manipulated
        :return: New signal object with transformed values
        """

    def _prepend_transform_name(self, string):
        return self.name.value + "_" + string


class TransformSeq(ABC):
    """Transform is a common interface for all transforms objects. Such
    objects manipulate a signal (e.g., applying log scaling, extracting
    MFCCs).
    Attrs:
        - name: The name of the transforms
    """

    def __init__(self, name: TransformType):
        self.name = name
        

    @abstractmethod
    def process(self, signals: list) -> list:
        """This method is responsible to apply a transforms to the incoming
        signals.
        :param signals: list of Signal objects to be manipulated
        :return: New list of signal objects with transformed values
        """

    def _prepend_transform_name(self, string):
        return self.name.value + "_" + string



class TransformSeq2Signal(ABC):

    def __init__(self, name: TransformType):
        self.name = name
    

    @abstractmethod
    def process(self, signals: list) -> list:
        """This method is responsible to apply a transforms to the incoming
        signals.
        :param signals: list of Signal objects to be manipulated
        :return: New signal objects with transformed values
        """

    def _prepend_transform_name(self, string):
        return self.name.value + "_" + string
