"""
This module contains a class that instantiates a FilePreprocessor object
from configurations.
"""

import logging
from typing import Dict
from radiomixer.io.loader.loaders import ClassicLoader, TIMITLoader
from radiomixer.io.saver.savers import WaveFeaturesSaver
from radiomixer.preprocessors.filepreprocessor import FilePreprocessor
from radiomixer.creator.transformchaincreator import TransformsChainCreator
from radiomixer.sampler.filesampler import FileSampler

logger = logging.getLogger(__name__)

class FilePreprocessorCreator:
    """FileProcessorCreator instantiates a FilePreprocessor object from config

    It offloads the creation of the transforms chain object to
    TransformsChainCreator.
    Attributes:
        - transforms_chain_creator: Instantiate a TransformsChain
    """

    def __init__(self, transforms_chain_creator: TransformsChainCreator):
        self.transforms_chain_creator = transforms_chain_creator

        # In case of adding new loaders/savers add key: value pair
        self.loader_types = {
            "ClassicLoader": ClassicLoader,
            "TIMITLoader": TIMITLoader,
            }
        self.saver_types = {
            "WaveFeaturesSaver": WaveFeaturesSaver
            }

        logger.info("Instantiated FileProcessorCreator object")

    def create(self, configs: Dict[str, dict]) -> FilePreprocessor:
        """Create a file preprocessor from transform configurations.

        The Loader and the Saver are instantiated in a hardcoded way. 
       

        :param configs: Dictionary of dictionary. Each nested dict provides
            configurations for loader and transforms chain. Configs example:
        :return: Instantiated file preprocessor
        """
        print(configs)
        fileSampler = FileSampler(configs=configs["FileSampler"])
    
        loader_module = self.loader_types.get(configs["FileLoader"]["type"])
        saver_module = self.saver_types.get(configs["FileSaver"]["type"])

        loader = loader_module(cfg = configs["FileLoader"]["configs"]) 
        transforms_chain = self.transforms_chain_creator.create(configs["transform_chain"])
        saver = saver_module(cfg = configs["FileSaver"]["configs"]) 
        

        file_preprocessor = FilePreprocessor(fileSampler = fileSampler,
                                             loader = loader,
                                             transforms_chain = transforms_chain,
                                             saver = saver)
        return file_preprocessor