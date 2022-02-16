"""
This module contains BatchFilePreprocessor, an object responsible to
preprocess all files in a path.
"""

import os
import logging

from radiomixer.preprocessors.filepreprocessor import FilePreprocessor
from radiomixer.utils.utils import create_dir_hierarchy_from_file

logger = logging.getLogger(__name__)

class SequentialFilePreprocessor:
    """
    
    SequentialFilePreprocessor preprocesses
    all files in a directory recursively and stores them on disk.
    It's a wrapper around a file processor that handles multiple files.
    Attributes:
        - preprocessor: Preprocess single file
        - configs:
            - num_files_generate: Number of files to be generated
            - save_dir: saving dirrectory
        
    """

    def __init__(self,
                 preprocessor: FilePreprocessor,
                 configs: dict):
        self.preprocessor = preprocessor
        self.num_files_generate = configs["num_files_generate"]
        self.save_dir = configs["save_dir"]
        logger.info("Instantiated SequentialFilePreprocessor object")
        

    def preprocess(self):
        """Batch preprocess all data in a dir recursively."""
        for idx in range(self.num_files_generate):
            save_path = self._infer_save_path(idx)
            create_dir_hierarchy_from_file(save_path)
            self.preprocessor.preprocess(save_path)
            if idx%1000==0:
                print(idx)
        

    def _infer_save_path(self, idx: int) -> str:
        save_path = os.path.join(self.save_dir, str(idx))
        return save_path