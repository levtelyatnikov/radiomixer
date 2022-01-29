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
Configuration file is divided into five sections:

- FileLoader
    - type
    - configs

- FileSaver
    - type
    - configs
  
- FileSampler
    - dataset_split
    - test_size
    - dataset_dirs
    - dataset_names
    - min_datasets
    - dataset_prob
    - replace
    - seed

- SequentialFileProcessor
    - num_files_generate
    - save_dir
    - save_config_dir

- transform_chain
    - Transforms 




The core of the library is the *preprocess* entry point. This script works 
with a config file. You set the type of preprocessing you want to apply in a 
yaml file, and then run the script. Your dataset will be entirely 
preprocessed and the results recursively stored in a directory of your 
choice that can potentially be created from scratch.

To run the entry point, ensure the library is installed and then type:
```shell
$ preprocess /path/to/config.yml
```

In the config.yml, you should provide the following parameters:
- `dataset_dir`: Path to the directory where your audio dataset is stored
- `save_dir`: Path where to save the preprocessed audio.
- Under `file_preprocessor`, you should provide settings for `loader` and `transforms_chain`.
- `loader`: Provide settings for the loader.
- `transforms_chain`: Parameters for each transform in the sequence. 
  of transforms which are applied to your data (i.e., TransformChain).

These config parameters are used to dinamically initialise the relative 
objects in the library. To learn what parameters are available at each 
level in the config file, please refer to the docstrings in the relative 
objects.

Check out `test/config.sampleconfig.yml` to see an example of a valid config 
file.

### Dataset list
[MUSAN](http://www.openslr.org/17/)
[GTZAN music-speech](http://marsyas.info/downloads/datasets.html)
[GTZAN Genre collection](http://marsyas.info/downloads/datasets.html)
[Scheirer & Slaney](https://labrosa.ee.columbia.edu/sounds/musp/scheislan.html)
[Instrument Recognition in Musical Audio Signals (https://www.upf.edu/web/mtg/irmas#:~:text=IRMAS%20is%20intended%20to%20be,violin%2C%20and%20human%20singing%20voice.)
[Singing Voice dataset](http://isophonics.net/SingingVoiceDataset), and  [LibriSpeech](http://www.openslr.org/12/).
