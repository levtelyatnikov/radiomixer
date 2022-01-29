"""
This module contains facilities to store numpy arrays. It can be used to
store audio features (e.g., MFCC, Spectrogram).
"""

import logging
import numpy as np

from radiomixer.io.signal import Signal
from radiomixer.io.saver.saver import Saver
from radiomixer.utils.utils import add_extension_to_file

logger = logging.getLogger(__name__)

class WaveFeaturesSaver(Saver):
    """WaveFeaturesSaver Signal object as npz files"""
    def __init__(self, cfg):
        super().__init__("npz")
        logger.info("Instantiated WaveFeaturesSaver object")

    def save(self,
             file: str,
             signal: Signal):
        """Store array as an npy file

        :param file: Path where to save file without extension
        :param array: Numpy array to store
        """
        file_with_extension = add_extension_to_file(file, self.extension)

        # Note: to get dict of parameters during loading 
        # It is necessary to index with [()]
        np.savez(file_with_extension,
                 audio = signal.data,
                 data_features = signal.data_features,
                 parameters = signal.parameters,
                 parameters_own = signal.parameters_own)
        