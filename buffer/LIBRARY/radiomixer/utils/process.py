import torch
import torchaudio

class AudioTransform():
    def __init__(self, **args):
        self.n_fft = args['n_fft']
        self.win_length = args['win_length']
        self.hop_length = args['hop_length']
        self.n_mels = args['n_mels']
        self.sample_rate = args['sample_rate']
        self.db = args['db']
        self.top_db = args['top_db']
        self.algo = args["algo"]
        self.device = args["device"]
        

        self.n_stft = self.n_fft//2 +1

        self.wav2mel = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_fft=self.n_fft, n_mels=self.n_mels,
                        win_length=self.win_length, hop_length=self.hop_length,
                        
                        # to be consistent with librosa
                        center=True,
                        pad_mode="reflect",
                        power=2.0,
                        norm="slaney",
                        onesided=True,
                        mel_scale="htk",
                        ).to(self.device)

        self.wav2spec = torchaudio.transforms.Spectrogram( 
                            n_fft=self.n_fft,
                            win_length=self.win_length,
                            hop_length=self.hop_length,
                            center=True,
                            pad_mode="reflect",
                            onesided=True,
                            power=2
                            ).to(self.device)
            
        self.mel2spec = torchaudio.transforms.InverseMelScale(n_stft=self.n_stft, n_mels=self.n_mels, sample_rate=self.sample_rate).to(self.device)
        self.griffinLim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length).to(self.device)
        self.spec2wav = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length).to(self.device)
        self.power2DB = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db).to(self.device)


        
                        
    def DB2power(self, x):
        return torchaudio.functional.DB_to_amplitude(x=x,ref=1.0, power=1)
    
    # WAV to Spectrograms
    def audio2spec(self, wav):
        return self.wav2spec(wav)

    def audio2mel(self, wav):
        M = self.wav2mel(wav)
        if self.db:
            M = self.power2DB(M)
        
        return M
    
    
    # Spectrograms to WAV
    def mel2wav(self, M):
        # db to power
        if self.db:
            M = self.DB2power(M)
        
        # mel spec to spec    
        spec = self.mel2spec(M)

        # spec to wav
        wav = self.spec2wav_f(spec)
        return wav

    def spec2wav_f(self, spec):
        # spec to wav
        if self.algo == 'griffinLim':
            wav = self.griffinLim(spec)
        else:
            wav = self.spec2wav(spec.to(dtype=torch.cdouble))
        return wav
        