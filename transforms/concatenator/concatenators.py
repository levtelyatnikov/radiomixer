import torch
from torch import Tensor

from radiomixer.io.signal import SignalFeature
from radiomixer.transforms.concatenator.concatenator import Concatenator
from radiomixer.transforms.transform import TransformType


class SummationConcatenator(Concatenator):
    """

    Sequentially concatenate diffetent segments
    with randomly sampled silence between the segments
    """

    def __init__(self):
        super().__init__(TransformType.SummationConcatenator)

    def _concatenate(self, signals: list) -> SignalFeature:
        """

        It is assumed that input signals have the same lengths
        hence current concatenation of the signal is simple 
        sumamtion. 
        """
        parameters, labels, files = [], [], []
        signal_out = torch.zeros(signals[-1].data.shape)
        
        for signal in signals:
            data = signal.data
            signal_out += data
            labels.append([signal.parameters["label"]] * data.shape[-1])
            parameters.append(signal.parameters)
            files.append(signal.file)
        

        signal = SignalFeature(sample_rate = signals[-1].sample_rate,
                                data = signal_out,                                
                                data_features = None,
                                file = files,
                                parameters = parameters,
                                parameters_own={
                                    "duration":signal_out.shape[1],
                                    "labels": labels}) #"
        return signal
    
# ---------------------------SequentialConcatenator---------------------------

class SequentialConcatenator(Concatenator):
    """

    Sequentially concatenate diffetent segment 
    with randomly sampled silence between the segments
    """

    def __init__(self):
        super().__init__(TransformType.SEQUENTIALCONCATENATOR)


    def _concatenate(self, signals: list) -> SignalFeature:
        n_signals = len(signals)

        # generate silence
        silences_lengths = signals[-1].parameters['silences_lengths']
        silences = [torch.zeros((1,l)) for l in silences_lengths]

        # generate final audio and labels
        data, labels = [], []
        for idx, signal in enumerate(signals):
            data.append(signal.data)
            labels.extend([signal.parameters['label']]*signal.data.shape[1])

            if idx == n_signals-1:
                pass
            else:
                data.append(silences[idx])
                labels.extend([0]*silences[idx].shape[1])

        return self._create_Signal(data = data,
                                   labels = labels,
                                   signals = signals)

    def _create_Signal(self, data: Tensor, labels: list , signals:list) -> SignalFeature:
        """

        Create SignalFeature object with
        all infromation from which final
        audio were created.
        :data: Final audio
        :labels: Labels
        :signals: list of Signals from which final audio were crated
        """
        sample_rate = 0
        files, parameters = [], []

        for signal in signals:
            sample_rate += signal.sample_rate
            files.append(signal.file)
            parameters.append(signal.parameters)
    
        audio = torch.concat(data, dim=1)
        signal = SignalFeature(sample_rate = sample_rate/len(signals),
                                data = audio,
                                labels = labels,
                                data_features = None,
                                file = files,
                                parameters = parameters,
                                parameters_own={"duration":audio.shape[1]})
        return signal
    

