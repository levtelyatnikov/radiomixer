"""Abstract interface for storing feature arrays on disk."""

import logging
from abc import ABC, abstractmethod
from radiomixer.io.signal import Signal

logger = logging.getLogger(__name__)

class Saver(ABC):
    """Abstract class that provides an interface to store signals / feature

    Attributes:
        - extension: Save extension (e.g., "npy")
    """

    def __init__(self, extension: str):
        self.extension = extension
        logger.info("Instantiated Saver object")

    @abstractmethod
    def save(self,
             file: str,
             Signal: Signal):
        """Store data to disk.

        :param file: Path where to save file without extension
        :param Signal: Consist of data to store
        """
