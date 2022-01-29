"""This module consist of different loaders"""

import logging
import torchaudio

from radiomixer.io.signal import Signal
from radiomixer.io.loader.loader import Loader
from radiomixer.utils.utils import remove_extension_from_file, add_extension_to_file

logger = logging.getLogger(__name__)

class ClassicLoader(Loader):
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        logger.info("Instantiated ClassicLoader object")
    
    def load(self, file:str, label) -> Signal:
        """Load audio file and Signal object.
        
        Parameters:
          :file: Path to audio file to load
          :label: Label can be any type, lately passed as a value to the dict

        :return: Signal

        """

        self._raise_file_extension_error_if_file_extension_isnt_allowed(file)
        waveform, sample_rate = torchaudio.load(file, normalize=self.normalize)
        waveform, sample_rate = self.wavetransforms.process(waveform, sample_rate)
        assert sample_rate == self.sample_rate,  "Expected sampling rate do not coinside with file sample rate"

        signal = Signal(sample_rate = sample_rate,
                        data = waveform,
                        parameters={"init_duration":waveform.shape[1], "label": label},
                        file = file)

        return signal

    def seq_load(self, files:list, labels:list) -> list:
        return [self.load(file=file, label=label) for file, label in zip(files, labels)]

#--------------------------------TIMITLoader---------------------------------#
class TIMITLoader(Loader):
    """TIMITLoader loading phonems into Signal object"""
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        logger.info("Instantiated TIMITLoader object")
    
    def load(self, file: str, label):
        """Load data into Signal.

        Parameters:
          :file: Path to audio file to load
          :label: Label can be any type, lately passed as a value to the dict

        :return: Signal
        """
      
        file_noext = remove_extension_from_file(remove_extension_from_file(file))
        PHONEM_FILE = add_extension_to_file(file_noext, 'PHN')
        phonems = self.read_phonems(PHONEM_FILE)

        self._raise_file_extension_error_if_file_extension_isnt_allowed(file)
        waveform, sample_rate = torchaudio.load(file, normalize=self.normalize)
        waveform, sample_rate = self.wavetransforms.process(waveform, sample_rate)

        assert sample_rate == self.sample_rate, "Expected sampling rate do not coinside with file sample rate"
       

        signal = Signal(sample_rate = sample_rate,
                        data = waveform,
                        parameters={"init_duration":waveform.shape[1], "label": label, 'phonems': phonems},
                        file = file)

        return signal
    
    def read_phonems(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(' ') for line in lines]
            phonems = [[(int(line[0]), int(line[1])), line[-1]] for line in lines]
        return phonems


