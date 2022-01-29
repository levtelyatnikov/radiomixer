
import torchaudio
from radiomixer.io.signal import SignalFeature
from radiomixer.transforms.transform import Transform, TransformType


class MelSpectrogram(Transform):
    """

    This class aims to modify the audio
    with help of torchaudio.transforms.Fade
    method. See the documentation for the set
    of available _parameters dict
    """

    def __init__(self, configs):
        super().__init__(TransformType.MELSPECTROGRAM)
        self._parameters = None

        self.n_fft = configs['n_fft']
        self.win_length = configs['win_length']
        self.hop_length = configs['hop_length']
        self.n_mels = configs['n_mels']
        self.sample_rate = configs['sample_rate']
        self.top_db = configs['top_db']
        self.db = configs['db']
        self.x_output_size = configs['x_output_size']


        self.wav2mel = torchaudio.transforms.MelSpectrogram(
                        sample_rate = self.sample_rate,
                        n_fft = self.n_fft, n_mels = self.n_mels,
                        win_length = self.win_length, hop_length = self.hop_length,
                        
                        # to be consistent with librosa
                        center=True,
                        pad_mode="reflect",
                        power=2.0,
                        norm="slaney",
                        onesided=True,
                        mel_scale="htk",
                        )
        if self.db:
            self.power2DB = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)
    
    def process(self, signal: SignalFeature) -> SignalFeature:
        signal.name = self._prepend_transform_name(signal.name)
        signal.data_features = self.wav2mel(signal.data)[:,:,:self.x_output_size]
        if self.db:
            signal.data_features = self.power2DB(signal.data_features)
        return signal