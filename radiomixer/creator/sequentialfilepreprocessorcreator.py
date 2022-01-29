"""
This module contains a class that instantiates a BatchFilePreprocessorCreator
object from configurations.
"""

import logging

from radiomixer.creator.transformfacory import TransformFactory
from radiomixer.creator.transformchaincreator import TransformsChainCreator
from radiomixer.creator.filepreprocessorcreator import FilePreprocessorCreator
from radiomixer.preprocessors.sequentialfileprocessor import SequentialFilePreprocessor

logger = logging.getLogger(__name__)

class BatchFilePreprocessorCreator:
    """BatchFilePreprocessorCreator instantiates a BatchFilePreprocessor

    It offloads the creation of the FilePreprocessor object to
    FilePreprocessorChainCreator.
    Attributes:
        - file_preprocessor_creator: Instantiate a FilePreprocessor
    """

    def __init__(self, file_preprocessor_creator: FilePreprocessorCreator):
        self.file_preprocessor_creator = file_preprocessor_creator
        logger.info("Instantiated BatchFilePreprocessorCreator object")

    def create(self, configs: dict) -> SequentialFilePreprocessor:
        """Create a batch file preprocessor from configurations.

        :param configs: Nested dictionary which contains config
        :return: Instantiated batch file preprocessor
        """
        file_preprocessor = self.file_preprocessor_creator.create(configs)
        batch_file_preprocessor = SequentialFilePreprocessor(file_preprocessor,
                                                             configs["SequentialFileProcessor"],
                                                             )

        return batch_file_preprocessor


def create_batch_file_preprocessor_creator() -> BatchFilePreprocessorCreator:
    """Instantiate a new BatchFilePreprocessorCreator object.

    :return: New batch file preprocessor creator
    """
    transform_factory = TransformFactory()
    transforms_chain_creator = TransformsChainCreator(transform_factory)
    file_preprocessor_creator = FilePreprocessorCreator(transforms_chain_creator)
    return BatchFilePreprocessorCreator(file_preprocessor_creator)