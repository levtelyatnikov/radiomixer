"""
This module provides a data structure that stores an audio signal with
additional info.
"""

import torch
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Signal:
    """Signal is a data structure that represents a signal.
    
    It can be used for storing waveforms, as well as other audio features
    E.g. transition parameters, labels
    Attributes:
        - name: Initially represent type of data. 
        Note: At the end of transforms consists of sequence applied transforms
        - sample_rate: Sampling rate of signal
        - data: Signal data
        - file: File path where signal was originally loaded from
        - parameters: Signal parameters
    """
    
    sample_rate: int
    data: torch.Tensor
    parameters: defaultdict
    file: str = "" 
    name: str = "waveform"
    
@dataclass
class SignalFeature:
    """Signal is a data structure that represents a signal.

    It can be used for storing waveforms, as well as other audio features
    E.g. transition parameters, labels, audio features (e.g., MFCC,
    MelSpectrogram).
    Attributes:
        - name: Consists of sequence applied transforms
        - sample_rate: Sampling rate of signal
        - data: Signal data
        - data_features: Features extracted from data
        - file: List of paths from which final audio were created
        - parameters: List of Signals parameters from which data were created
        - parameters_own: Parameters of current signal in data
    """
    
    sample_rate: int
    data: torch.Tensor 
    data_features: torch.Tensor 
    #labels: list
    parameters: list 
    parameters_own: defaultdict
    file: list 
    name: str = "waveform" 

    


