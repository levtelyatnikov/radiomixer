
import torchaudio
from pathlib import Path




class FileExtensionError(Exception):
    """Error that is thrown when the extension of a file isn't allowed."""

def extract_extension_from_file(file: str) -> str:
    """Extract extension from file name.
    :param file: Path of file we want to extract the extension from.
    :return: Extension of file (e.g., mp3)
    """
    return Path(file).suffix.lower()[1:]


class WaveformManipulation():
    def __init__(self, configs):
        self.sample_rate = configs["sample_rate"]
        self.remove_silence_bool = configs["remove_silence"]
        self.min_duration = self.sample_rate * configs["min_duration"]
        
    
    def remove_silence(self, waveform, sample_rate):
        if self.remove_silence_bool:
            waveform = torchaudio.functional.vad(waveform, sample_rate=sample_rate)
        return waveform

    def enlarge_duration(self, waveform):
        duration = waveform.shape[1]
        if duration < self.min_duration:
            repeat = self.min_duration//duration + 1
            waveform = waveform.repeat(1, repeat)
        return waveform

    def resample(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resample(waveform)
        return waveform

    def process(self, waveform, sample_rate):
        waveform = self.remove_silence(waveform, sample_rate)
        waveform = self.resample(waveform, sample_rate)
        waveform = self.enlarge_duration(waveform)
        return waveform, self.sample_rate
        





        

