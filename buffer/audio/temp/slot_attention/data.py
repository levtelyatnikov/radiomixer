import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from omegaconf.dictconfig import DictConfig

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from slot_attention.utils import compact, rescale


class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        if max_num_images == -1:
            self.max_num_images = None
        else:
            self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "images", split)
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(self, index: int):
        image_path = self.files[index]
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.clevr_transforms(img)

    def __len__(self):
        return len(self.files)

    def get_files(self) -> List[str]:
        with open(os.path.join(self.data_root, f"scenes/CLEVR_{self.split}_scenes.json")) as f:
            scene = json.load(f)
        paths: List[Optional[str]] = []
        total_num_images = len(scene["scenes"])
        i = 0
        while (self.max_num_images is None or len(paths) < self.max_num_images) and i < total_num_images:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            if num_objects_in_scene <= self.max_n_objects:
                image_path = os.path.join(self.data_path, scene["scenes"][i]["image_filename"])
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)
            i += 1
        return sorted(compact(paths))


class CLEVRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clevr_transforms: Callable,
        max_n_objects: int,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = CLEVRDataset(
            data_root=self.data_root,
            max_num_images=self.num_train_images,
            clevr_transforms=self.clevr_transforms,
            split="train",
            max_n_objects=self.max_n_objects,
        )
        self.val_dataset = CLEVRDataset(
            data_root=self.data_root,
            max_num_images=self.num_val_images,
            clevr_transforms=self.clevr_transforms,
            split="val",
            max_n_objects=self.max_n_objects,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CLEVRTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)




import torch, torchaudio
from pathlib import Path
import pandas as pd
import glob 
import os

###########################################################################################
###################################### Synth Hard dataset #################################
###########################################################################################

class MusanSynth(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self, transforms,  path, sample_rate: int = 22050,seed = 10):
        np.random.seed(seed)
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        
        paths = glob.glob(os.path.join(self.path,"*.npy"), recursive=True)

        self.labels_paths = np.array(sorted([ path for path in paths if "label" in path ]))
        self.mel_paths = np.array(sorted([ path for path in paths if "label" not in path ]))


    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        PATH = self.mel_paths[index]
        #LABEL_MEL = self.labels_paths[index]
        
        # load and normalize
        M = torch.Tensor(np.load(PATH)[np.newaxis,:]/80)
        #label = np.load(LABEL_MEL)
        assert list(M.shape) == [1,128,128]
        
        return M
        
    def __len__(self):
        # Returns length
        return len(self.mel_paths)


class SyntaticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()
        
        self.train_batch_size = cfg.dataset.train_batch_size
        self.val_batch_size = cfg.dataset.val_batch_size
        self.num_workers = cfg.dataset.num_workers
        
        num_train_images = cfg.dataset.num_train_images
        num_val_images = cfg.dataset.num_val_images 
        
        seed = cfg.dataset.seed


        train_path = "/home/lev/datasets/musan/synth_mel_train"
        val_path = "/home/lev/datasets/musan/synth_mel_val"
        # Load data
        self.train_dataset = MusanSynth(transforms=None,path = train_path)
        self.val_dataset = MusanSynth(transforms=None, path = val_path)
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

###########################################################################
###########################################################################
###########################################################################

class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self,transforms,  path: Path, 
                 sample_rate: int = 16000,
                 folds = [1]):
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        self.csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        
        audio_dur = 5
        hop = sample_rate* audio_dur//128
        fft_lenght = 2*hop
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=fft_lenght, hop_length=hop)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        self.transforms = transforms


    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / Path('audio') / Path(row['filename']))
        label = row['target']

        wav = self.resample(wav)
        
        xb = self.transforms(self.db(
            self.melspec(wav)))[:,:128,:128]
        
        assert list(xb.shape) == [1,128,128]
        xb = rescale(xb)
        return xb
        
    def __len__(self):
        # Returns length
        return len(self.csv)
    

class ESC50DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, clevr_transforms):
        super().__init__()

        self.data_root = cfg.dataset.data_root,
        self.train_batch_size = cfg.dataset.batch_size,
        self.val_batch_size = cfg.dataset.val_batch_size,
        self.clevr_transforms = clevr_transforms, 
        self.num_workers = cfg.dataset.num_workers

        # Load data
        self.train_dataset = ESC50Dataset(path=self.data_root, transforms=self.clevr_transforms, folds=[1,2,3,5])#
        self.val_dataset = ESC50Dataset(path=self.data_root, transforms=self.clevr_transforms, folds=[4])
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


import numpy as np
class SyntaticDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self, t: int = 5, sr: int = 8000,
                num_audio: int = 10000, freq_n: int=5,
                db = True, name: str='None', freq_range: int=5000,
                seed = 10,
                ):
        np.random.seed(seed)
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.t, self.sr = t, sr
        self.freq_n = freq_n
        self.db_bool = db
        self.name = name
        self.num_audio = num_audio
        self.freq_range = np.arange(freq_range)
        self.ns = np.arange(1, self.freq_n+1)
        
        # Prepare mel spec
        self.time_disk = np.arange(self.t*self.sr)
        hop = self.sr* self.t//128
        fft_lenght = 2*hop
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=fft_lenght, hop_length=hop)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.hard_data_iter = 2
        # Generate data
        self.generate_data()
    
    
    def generate_audio_easy(self, freqs):
        return np.array([np.sin(2*np.pi*f*self.time_disk/self.sr) for f in freqs]).sum(axis=0)

    def generate_audio_mid(self, freqs):
        step = self.time_disk.shape[0]//len(freqs)
        return np.concatenate([np.sin(2*np.pi*f*self.time_disk[i*step:(step+1)*(i+1)]/self.sr) for i,f in enumerate(freqs)])[:self.time_disk.shape[0]]

    def normalize_signal(self, xb):
        xb -= xb.min(1, keepdim=True)[0]
        xb /= xb.max(1, keepdim=True)[0]+1e-10
        return xb
    
    def signal_stat(self):
        # choose number of freq
        n = np.random.choice(self.ns, size=1)
        # return n freq
        return np.random.choice(self.freq_range, size=n, replace=False) 
    
    def get_melspec(self, signal):
        if self.db_bool==True:
            xb = self.db(self.melspec(signal.unsqueeze(0)))[:, :128, :128]
        else: 
            xb = self.melspec(signal.unsqueeze(0))[:, :128, :128]
        return self.normalize_signal(xb)

    def easy_data(self,):
        for _ in range(self.num_audio):
            freqs = self.signal_stat()
            # signal with constant tones (mel spec with continious lines) 
            signal = torch.Tensor(self.generate_audio_easy(freqs))
            xb = self.get_melspec(signal)
            assert list(xb.shape) == [1,128,128]
            self.data.append(xb)

    def mid_data(self,):
        for _ in range(self.num_audio):
            freqs = self.signal_stat()
            signal = torch.Tensor(self.generate_audio_mid(freqs))
            xb = self.get_melspec(signal)
            assert list(xb.shape) == [1,128,128]
            self.data.append(xb)  
    
    def  hard_data(self,):
        for _ in range(self.num_audio):
            freqs = self.signal_stat()
            signal = torch.Tensor(self.generate_audio_mid(freqs))

            for _ in range(self.hard_data_iter): 
                freqs = self.signal_stat()
                signal += torch.Tensor(self.generate_audio_mid(freqs))

            xb = self.get_melspec(signal)
            assert list(xb.shape) == [1,128,128]
            self.data.append(xb) 

    def generate_data(self):
        self.data = []
        if self.name == 'easy_dataset':
            self.easy_data()

        elif self.name == 'mid_dataset':
            self.mid_data()

        elif self.name == 'hard_dataset':
            self.hard_data()
        
    def __getitem__(self, index):
        return  self.data[index]
        
    def __len__(self):
        # Returns length
        return self.num_audio


class SyntaticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()
        
        self.train_batch_size = cfg.dataset.train_batch_size
        self.val_batch_size = cfg.dataset.val_batch_size
        self.num_workers = cfg.dataset.num_workers

        t = cfg.dataset.time
        sr = cfg.dataset.sampling_rate 
        num_train_images = cfg.dataset.num_train_images
        num_val_images = cfg.dataset.num_val_images 
        freq_n = cfg.dataset.freq_n 
        db = cfg.dataset.db
        name = cfg.dataset.name
        freq_range = cfg.dataset.max_freq # in fact max freq have to be sr//2
        seed = cfg.dataset.seed

        # Load data
        self.train_dataset = SyntaticDataset(t=t, sr=sr, num_audio=num_train_images,
                                            freq_n=freq_n, db = db, name=name, 
                                            freq_range=freq_range, seed=seed)
        self.val_dataset = SyntaticDataset(t=t, sr=sr, num_audio=num_val_images,
                                            freq_n=freq_n, db = db, name=name, 
                                            freq_range=freq_range, seed=10)
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
