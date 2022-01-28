"""
This module contains a class that instantiates a transforms chain object
from configuration for each transform in the chain.
"""
import logging
from typing import Dict, List

from radiomixer.creator.transformfacory import TransformFactory
from radiomixer.creator.transformchain import TransformsChain
from radiomixer.transforms.transform import Transform

logger = logging.getLogger(__name__)

class TransformsChainCreator:
    """
    
    TransformsChainCreator instantiates a TransformsChain object from
    config, offloading the creation of concrete transforms to a transform
    factory.

    Attributes:
        - transform_factory (TransformFactory): Factory that instantiates
            transforms
    """

    def __init__(self, transform_factory: TransformFactory):
        self.transform_factory = transform_factory
        logger.info("Initialised TransformsChainCreator object")

    def create(self, configs: Dict[str, dict]) -> TransformsChain:
        """Create a transforms chain from transform configurations.
        
        Check out documentation to check all the
        available arguments to pass in the configs.
        :return: Instantiated transforms chain
        """
        transforms = self._create_transforms(configs)
        transforms_chain = TransformsChain(transforms)
        return transforms_chain

    def _create_transforms(self, configs: Dict[str, dict]) -> List[Transform]:
        transforms = []
        for transform_type, transform_config in configs.items():
            
            transform = self.transform_factory.create(transform_type,
                                                      **transform_config)
            transforms.append(transform)
        return transforms