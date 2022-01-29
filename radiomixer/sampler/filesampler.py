import os
import glob
import numpy as np
import itertools
from sklearn.model_selection import train_test_split



class FileSampler():
    def __init__(self, configs: dict): 

        """

        FileSampler is responsible for providing filepath 
        to generate new audios.
        Arguments:
            dataset_dirs,
            dataset_names,
            seed = 0,
            dataset_prob=None, replace=False
            
        """
        np.random.seed(configs['seed'])
        self.dataset_paths = configs['dataset_dirs'] 
        self.dataset_prob = configs['dataset_prob']
        self.replace = configs['replace'] 
        self.dataset_names = configs['dataset_names']

        self.dataset_split = configs['dataset_split']
        self.test_size = configs['test_size']
        self.seed = configs['seed']

        self.min_datasets = configs['min_datasets']
        self.n_datasets = len(self.dataset_names)

        # create dict of "dataset name : audio_paths"
        
        #datasetAudioDict = dict(zip(self.dataset_names, list(map(lambda x: glob.glob(os.path.join(x, "**/*.wav"), recursive=True)), self.dataset_paths)))
        self.datasetAudioDict = self.generate_split()

    def generate_split(self,):
        datasetAudioDict = dict(zip(self.dataset_names, [sorted(glob.glob(os.path.join(x, "**/*.wav"), recursive=True)) for x in self.dataset_paths]))
        train_data = {}
        test_data = {}

        for key, value in datasetAudioDict.items():
            
                train, test = train_test_split(value,
                                test_size = self.test_size,
                                random_state = self.seed)
             
                train_data[key] = train
                test_data[key] = test

        # double check

        a = set(itertools.chain(*[train_data[key] for key in train_data.keys()])) #set(itertools.chain(train_data[key] for key in train_data.keys()))
        b = set(itertools.chain(*[test_data[key] for key in test_data.keys()])) 
        assert len(a.intersection(b)) == 0, f"Train Test leak {len(a.intersection(b))}"

        if self.dataset_split == "train":
            return train_data

        else:
            return test_data
            
                
            
    
    def sample_files(self):
        """

        Sample the datasets from which 
        Futher audios will be retreived

        parameters logic:
        p: allows to cotrol type of audios sampled
        replace: allows to avoid self transition

        Return: list of sampled files, datasets list
        """
        
        size = np.random.randint(self.min_datasets, high = self.n_datasets + 1, size = 1, dtype = int)
        datasets = np.random.choice(self.dataset_names,
                                    size = size,
                                    replace=self.replace,
                                    p = self.dataset_prob)
                                    
        files = np.array([np.random.choice(self.datasetAudioDict[dataset], size=1)[0] for dataset in datasets])
        
        return files, datasets
