"""
This module contains a class that applies multiple transforms
sequentially on a signal.
"""

import logging
from typing import List
from radiomixer.transforms.transform import Transform
from radiomixer.io.signal import SignalFeature

logger = logging.getLogger(__name__)

class TransformsChain():
    """Apply multiple transforms on a signal in a sequential manner."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
        logger.info("Initialised TransformsChain object")

    @property
    def transforms_names(self):
        transform_names = [transform.name.value for transform in
                           self.transforms]
        return transform_names

    def process(self, x: list) -> SignalFeature:
        """Apply multiple transforms sequentially to a signal.

        :param signal: Signal to transform
        :return: Modified signal
        """
        for transform in self.transforms:
            x = transform.process(x)
        return x



 