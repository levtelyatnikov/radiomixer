"""Abstract interface for extracting and applying filter to signal"""


import logging
from abc import abstractmethod
from radiomixer.io.signal import Signal
from radiomixer.transforms.transform import TransformSeq2Seq, TransformType

logger = logging.getLogger(__name__)

class Mixer(TransformSeq2Seq):
    """Abstract class to provide interface to extract and appply transitions


    Mixer class is responsible two main attributes: 
      - Extract segment from audio
      - Apply generated transitions
      - Optionally normalize audio after transitions

    Attributes:
        - name: TransformType name which then passed into TransformSeq
    """

    def __init__(self, name: TransformType):
        super().__init__(name)
        logger.info("Instantiated Saver object")

    @abstractmethod
    def _pocess(self, signal: Signal) -> Signal:
        """This func. is responsible for applying sequence of transforms. 

        - Extract
        - Fade
        - optional Normalize
        """

    def process(self, signals):
        return  [self._pocess(signal) for signal in signals]

