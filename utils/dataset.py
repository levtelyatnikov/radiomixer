import os
import glob
import numpy as np

import torch 
import torchaudio



class AudioDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self,  dataset_paths, dataset_names,
                 sample_rate: int = 22050, min_duration=1,
                 remove_silence=True,
                 normalize=False):
        """

        root_paths: paths to folders with .wav files. 
        Ex: [[<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>], [<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>]]
        
        sample_rate: default is 22050
        normalize: normalize audio uploaded audio into [-1 ,1] range
        min_duration: minimum duration in seconds 
        """
        self.normalize = normalize
        self.sample_rate = sample_rate 
        self.remove_silence = remove_silence
        self.min_duration = self.sample_rate*min_duration
        self.unique_dataset_names = dataset_names
        

        self.audio_paths = list(map(lambda x: glob.glob(os.path.join(x,"**/*.wav"), recursive=True), dataset_paths))
        
        self.dataset_names = np.concatenate([len(paths)*[name] for name, paths  in zip(self.unique_dataset_names, self.audio_paths)])
        self.audio_paths = np.concatenate(self.audio_paths)


    def __getitem__(self, index):
        PATH, NAME = self.audio_paths[index], self.dataset_names[index]
        waveform, sample_rate = torchaudio.load(PATH, normalize=self.normalize) # normalize to map waveform into [-1,1]

        # check input audio consistent with default sr 
        assert sample_rate == self.sample_rate
        
        if self.remove_silence:
            waveform = self.delete_silence(waveform)
        
        duration = waveform.shape[1]
        if duration < self.min_duration:
            waveform = waveform.repeat(1, np.ceil(duration//self.min_duration))

        return waveform, NAME
class AudioDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self,  dataset_paths, dataset_names,
                 sample_rate: int = 22050, min_duration=1,
                 remove_silence=True,
                 normalize=False):
        """

        root_paths: paths to folders with .wav files. 
        Ex: [[<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>], [<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>]]
        
        sample_rate: default is 22050
        normalize: normalize audio uploaded audio into [-1 ,1] range
        min_duration: minimum duration in seconds 
        """
        self.normalize = normalize
        self.sample_rate = sample_rate 
        self.remove_silence = remove_silence
        self.min_duration = self.sample_rate*min_duration
        self.unique_dataset_names = dataset_names
        

        self.audio_paths = list(map(lambda x: glob.glob(os.path.join(x,"**/*.wav"), recursive=True), dataset_paths))
        
        self.dataset_names = np.concatenate([len(paths)*[name] for name, paths  in zip(self.unique_dataset_names, self.audio_paths)])
        self.audio_paths = np.concatenate(self.audio_paths)


    def __getitem__(self, index):
        PATH, NAME = self.audio_paths[index], self.dataset_names[index]
        waveform, sample_rate = torchaudio.load(PATH, normalize=self.normalize) # normalize to map waveform into [-1,1]

        # check input audio consistent with default sr 
        assert sample_rate == self.sample_rate
        
        if self.remove_silence:
            waveform = self.delete_silence(waveform)
        
        duration = waveform.shape[1]
        if duration < self.min_duration:
            waveform = waveform.repeat(1, np.ceil(duration//self.min_duration))

        return waveform, NAME
class AudioDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self,  dataset_paths, dataset_names,
                 sample_rate: int = 22050, min_duration=1,
                 remove_silence=True,
                 normalize=False):
        """

        root_paths: paths to folders with .wav files. 
        Ex: [[<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>], [<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>]]
        
        sample_rate: default is 22050
        normalize: normalize audio uploaded audio into [-1 ,1] range
        min_duration: minimum duration in seconds 
        """
        self.normalize = normalize
        self.sample_rate = sample_rate 
        self.remove_silence = remove_silence
        self.min_duration = self.sample_rate*min_duration
        self.unique_dataset_names = dataset_names
        

        self.audio_paths = list(map(lambda x: glob.glob(os.path.join(x,"**/*.wav"), recursive=True), dataset_paths))
        
        self.dataset_names = np.concatenate([len(paths)*[name] for name, paths  in zip(self.unique_dataset_names, self.audio_paths)])
        self.audio_paths = np.concatenate(self.audio_paths)


    def __getitem__(self, index):
        PATH, NAME = self.audio_paths[index], self.dataset_names[index]
        waveform, sample_rate = torchaudio.load(PATH, normalize=self.normalize) # normalize to map waveform into [-1,1]

        # check input audio consistent with default sr 
        assert sample_rate == self.sample_rate
        
        if self.remove_silence:
            waveform = self.delete_silence(waveform)
        
        duration = waveform.shape[1]
        if duration < self.min_duration:
            waveform = waveform.repeat(1, np.ceil(duration//self.min_duration))

        return waveform, NAME
class AudioDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self,  dataset_paths, dataset_names,
                 sample_rate: int = 22050, min_duration=1,
                 remove_silence=True,
                 normalize=False):
        """

        root_paths: paths to folders with .wav files. 
        Ex: [[<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>], [<PATH TO DATASET/AUDIO TYPE FOLDER/*.WAV>]]
        
        sample_rate: default is 22050
        normalize: normalize audio uploaded audio into [-1 ,1] range
        min_duration: minimum duration in seconds 
        """
        self.normalize = normalize
        self.sample_rate = sample_rate 
        self.remove_silence = remove_silence
        self.min_duration = self.sample_rate*min_duration
        self.unique_dataset_names = dataset_names
        

        self.audio_paths = list(map(lambda x: glob.glob(os.path.join(x,"**/*.wav"), recursive=True), dataset_paths))
        
        self.dataset_names = np.concatenate([len(paths)*[name] for name, paths  in zip(self.unique_dataset_names, self.audio_paths)])
        self.audio_paths = np.concatenate(self.audio_paths)


    def __getitem__(self, index):
        PATH, NAME = self.audio_paths[index], self.dataset_names[index]
        waveform, sample_rate = torchaudio.load(PATH, normalize=self.normalize) # normalize to map waveform into [-1,1]

        # check input audio consistent with default sr 
        assert sample_rate == self.sample_rate
        
        if self.remove_silence:
            waveform = self.delete_silence(waveform)
        
        duration = waveform.shape[1]
        if duration < self.min_duration:
            waveform = waveform.repeat(1, np.ceil(duration//self.min_duration))

        return waveform, NAME
        
    def __len__(self):
        # Returns length
        return len(self.audio_paths)
    
    def delete_silence(self, wav):
        return torchaudio.functional.vad(wav, sample_rate=self.sample_rate)

# resample 
#https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic/commit/e904c32bfe772fba4b068067490e6b2b98bb9c4a

# dataset_paths = [  '/home/lev/datasets/musan/music',
#                 '/home/lev/datasets/musan/speech'
#                 ]
# dataset_names = ['music', 'speech']

# audio_dataset = AudioDataset(dataset_paths=dataset_paths,dataset_names=dataset_names)