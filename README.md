# radiomixer

<p align="center">
    <a href="https://docs.anaconda.com/"><img alt="PyTorch" src="https://anaconda.org/conda-forge/librosa/badges/version.svg"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
</p>

![alt text](https://github.com/levtelyatnikov/radiomixer/blob/main/MEM.png)


# Structure

```bash
.
├── configs                # yaml files to generate different datasets 
│   ├── teamplate1.yaml    # teamplate1 dataset configuration
│   └── teamplate2.yaml    # teamplate2 dataset configuration
├── requirements.txt    # basic requirements
├── radiomixer
│   ├── config          # Configuration loader/saver/validator modules
│   ├── creator         # PyTorch Lightning datamodules and datasets
│   ├── io              # Audio file loader/saver + Signal datatype module
│   ├── preprocessors   # Single/Sequential file processor logic
│   ├── sampler         # Transition sampler + Audio segment sampler
│   ├── transforms      # Transforms (e.g. audio concatenator, filters, mixer, feature extractors, scallers)
│   └── utils           # Utils
└── notebooks           # Example notebook

```
# RadioMixer
This repository contains code associated with the paper titled "Artificially Synthesising Data for Audio Classification and Segmentation to Improve Speech and Music Detection in Radio Broadcast" accepted for publication in IEEE ICASSP 2021 [[PDF](https://arxiv.org/pdf/2102.09959.pdf)].

Machine learning models for audio segmentation music-speech detection are generally trained on proprietary audio, which cannot be shared. More importantly, current **self-supervised methods in the audio domain require novel audio augmentation with reliable labels to perform auxiliary tasks.**
RadioMixer is the tool that will allow you:
 -  Mix audios with radio DJ workflow. 
 -  Extract audio features

## How do I augment an audio datasets? ##
To perform datasets augmentation it is necessary to do a cpoule of steps:
 - Download and store initial datasets
 - Configure configuration file (e.g. teamplate1.yaml)
 - Run the script

### Download and store initial datasets
Download and store your datasets on your local machine. There is no possibility to process datasets with different sampling rates. However, it will be done in the near future.

### Configuration file
#### Code structure
TransformerChain is in charge of sequentially applying transformations/manipulations to the input audio(s). TransformerChain is configured through .yaml file (see [Configuration logic](#configuration-logic)). The package functionality is divided into different parts:
- File loader
- File Saver
- File Sampler
- TransformerChain


Entry point of the code is a preprocessor (SequentialFileProcessor object is configured in configuration file.)

If you want to understand better the code organization please refer to [praudio](https://github.com/musikalkemist/praudio)
#### Configuration file logic
This section will provied with a brief description of the configuration file. Take a look at **teamplate1.yaml** for more details
Configuration file is divided into five sections:
- FileLoader
    - `type`:    Loader type
    - `configs`: Loader configs
- FileSaver
    - `type`:    Saver type
    - `configs`: Saver configs
- FileSampler
    - `dataset_split`:  Train/test split (which part of the dataset is foing to be generated)
    - `test_size`:      Test size
    - `dataset_dirs`:   Dataset dirrectories in a list (e.g ["dataset1/", "dataset2/"...])
    - `dataset_names`:  Dataset names corresponding to dataset dirrectories
    - `min_datasets`:   Minimum number of dataset used to generate one audio
    - `dataset_prob`:   Samling dataset probability distribution 
    - `replace`:        Possibility to samle same dataset
    - `seed`:           Seed
- SequentialFileProcessor
    - `num_files_generate`: Number of files to generate
    - `save_dir`:           Save dirrectory
    - `save_config_dir`:    Config save dirrectory
- transform_chain
    - `Transforms`: Sequence of audio transforms and manipulations

### Organization
#### Signal object
#### Loader
Loader is responsible for loading an audio file and create Signal object.
Available modules:
 - `ClassicLoader`
 - `TIMITLoader`
See `io/loader/loaders.py` for parameters needed to be passed into modules

#### Saver
Saver is responsible for saving  Signal object as npz file
Available modules:
 - `WaveFeaturesSaver`
See `io/loader/savers.py` for parameters needed to be passed into modules

#### FileSampler
File sampler is responsible for providing filepaths of audio which will be loaded into Signal object and used to generate new (augmented) audio.

#### TransformChain
Transform Chain (TR) is responsible for applying different manipulation to audios. There are many steps for which TR is responsible:
- Generate segment parameters for audio segments from loaded audios. (e.g. initial timestamp, segment duration) (Available modules: `EqualSegmentSampler`)
- Mixer (Available modules: `CustomMixer`, `TorchMixer`). Mixer module is a composition of next manipulations:
    - Generate transiton parameters (Available modules: `TransitionOverlapedSegmentsParametersSampler`)
    - Extract audio segments (Available modules: `ExtractSegment`)
    - Apply Fade in/out manipulations (`CustomFilter`, `TorchFilterIn`, `TorchFilterOut`)
    

### Dataset list
[MUSAN](http://www.openslr.org/17/)

[GTZAN music-speech](http://marsyas.info/downloads/datasets.html)

[GTZAN Genre collection](http://marsyas.info/downloads/datasets.html)

[Scheirer & Slaney](https://labrosa.ee.columbia.edu/sounds/musp/scheislan.html)

[Instrument Recognition in Musical Audio Signals](https://www.upf.edu/web/mtg/irmas#:~:text=IRMAS%20is%20intended%20to%20be,violin%2C%20and%20human%20singing%20voice.)

[Singing Voice dataset](http://isophonics.net/SingingVoiceDataset), and  [LibriSpeech](http://www.openslr.org/12/)

[More datasets](https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad)
