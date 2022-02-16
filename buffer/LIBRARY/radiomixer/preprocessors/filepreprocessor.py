"""This module features a class that preprocesses an audio file in one go."""

import logging

from radiomixer.creator.transformchain import TransformsChain
from radiomixer.sampler.filesampler import FileSampler
from radiomixer.io.loader.loader import Loader
from radiomixer.io.saver.saver import Saver

logger = logging.getLogger(__name__)

class FilePreprocessor:
    """
    
    FilePreprocessor is responsible for performing 
    a full sequence of audio manipulations:
      - Sample files
      - Load files
      - Apply transformations
      - Save files
    
    Attributes:
        - fileSampler: File sampler object
        - loader: Loader object
        - transforms_chain: Sequence of transforms to apply to signal
        - saver: Object responsible for saving a transformed signal to disk
    """

    def __init__(self,
                 fileSampler: FileSampler,
                 loader: Loader,
                 transforms_chain: TransformsChain,
                 saver: Saver):
        self.fileSampler = fileSampler
        self.loader = loader
        self.transforms_chain = transforms_chain
        self.saver = saver
        logger.info("Instantiated FilePreprocessor object")
        
    def preprocess(self, save_path: str):
        """

        Preprocess an audio file with the followiing steps:
          - Sample files from which segments will be chosen
          - Load audio files into Signal class
          - Apply chain of transformations to signals
          - Store transformed signal

        :param save_path: Path where to save processed signal. 
        Note: The path shouldn't have an extension
        """
        file_paths, dataset_names = self.fileSampler.sample_files()
        signals = self.loader.seq_load(file_paths, dataset_names)
        signal = self.transforms_chain.process(signals)
        self.saver.save(save_path, signal)
        