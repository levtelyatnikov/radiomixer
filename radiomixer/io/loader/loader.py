import logging
from abc import ABC, abstractmethod

from radiomixer.io.signal import Signal
from radiomixer.io.loader.loader_utils import FileExtensionError,\
    WaveformManipulation, extract_extension_from_file


logger = logging.getLogger(__name__)

class Loader(ABC):
    """Abstract class that provides an interface to load signals (audio)"""

    def __init__(self,  configs: dict):
        """

        Loader class is responsible for:
          - loading the audio
          - delete silence if needed
          - normalization or standartization input signal

        Arguments: (default) 
            sample_rate: int
            min_duration: float
            remove_silence: bool
            normalize: bool
        """

        self.sample_rate = configs["sample_rate"]
        self.normalize = configs["normalize"]
        self.min_duration = self.sample_rate * configs["min_duration"]
        
        self.wavetransforms = WaveformManipulation(configs=configs)
        
        self._signal_type = "waveform"
        self._audio_file_extensions = [
            "wav", "wave", "mp3",
            "ogg", "flac"
        ]

        logger.info("Instantiated Loader object")

            
    @abstractmethod
    def load(self,
             file: str,
             label) -> Signal:
        """Load data into Signal.

        Parameters:
          :file: Path to audio file to load
          :label: Label can be any type, lately passed as a value to the dict

        :return: Signal
        """

    def seq_load(self, files:list, labels:list) -> list:
        return [self.load(file=file, label=label) for file, label in zip(files, labels) ]

    def _raise_file_extension_error_if_file_extension_isnt_allowed(self, file):
        extension = extract_extension_from_file(file)
        if extension not in self._audio_file_extensions:
            raise FileExtensionError(f"'{extension}' extension can't be loaded.")